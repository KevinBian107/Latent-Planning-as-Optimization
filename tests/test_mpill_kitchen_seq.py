import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import minari
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.MPILL import MPILearningLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

device = torch.device("cpu")

MAX_LEN = 50
HIDDEN_SIZE = 128
BATCH_SIZE = 16
N_EPOCHS = 50
ALPHA_P_MOMENTUM = 0.99  # Slow learning
KL_WEIGHT = 0.01  # Zeta VI
ALPHA_WEIGHT = 0.3  # TD-BU
LR = 1e-4
LR_ENCODER = 1e-4  # η_φ
LR_DECODER = 1e-4  # η_ψ
LR_TRAJ_GEN = 1e-4  # η_β
LR_REWARD = 1e-4   # η_γ

WINDOW_SIZE = 10  # Window size for moving average
SMOOTH_ALPHA = 0.05  # Exponential moving average smoothing factor

MICROWAVE_IDX = 31    # Index for microwave door position
KETTLE_IDX_X = 32     # Index for kettle x-coordinate
KETTLE_IDX_Y = 33     # Index for kettle y-coordinate
KETTLE_IDX_Z = 34     # Index for kettle z-coordinate
LIGHT_SWITCH_IDX = 26 # Index for light switch position
SLIDE_CABINET_IDX = 28 # Index for sliding cabinet door position

MICROWAVE_THRESHOLD = 0.2      # Threshold for considering microwave open
KETTLE_MOVE_THRESHOLD = 0.1    # Threshold for considering kettle moved
LIGHT_SWITCH_THRESHOLD = -0.6  # Threshold for considering light switch on
SLIDE_CABINET_THRESHOLD = 0.2  # Threshold for considering sliding cabinet open


def detect_subtasks(episode):
    """
    Detect which subtasks are completed in a trajectory
    
    Args:
        episode: Minari episode object containing observations
        
    Returns:
        list of str: Completed subtask IDs
    """
    # Get observation sequence
    observations = episode.observations["observation"]
    
    initial_state = observations[0]
    final_state = observations[-1]
    
    subtasks = []
    # Check if microwave is open
    if final_state[MICROWAVE_IDX] > MICROWAVE_THRESHOLD:
        subtasks.append("microwave")
    
    # Check if kettle has moved
    kettle_moved = np.linalg.norm(
        final_state[KETTLE_IDX_X:KETTLE_IDX_Z+1] - 
        initial_state[KETTLE_IDX_X:KETTLE_IDX_Z+1]
    ) > KETTLE_MOVE_THRESHOLD
    if kettle_moved:
        subtasks.append("kettle")
    
    # Check if light switch is on
    if final_state[LIGHT_SWITCH_IDX] < LIGHT_SWITCH_THRESHOLD:
        subtasks.append("light")
    
    # Check if sliding cabinet is open
    if final_state[SLIDE_CABINET_IDX] > SLIDE_CABINET_THRESHOLD:
        subtasks.append("slidecabinet")
    
    return subtasks


def determine_task_id(episode):
    """
    Determine task ID based on completed subtasks
    
    - There may be multiple subtask completed, depending on what the agent decides to do.
    """
    try:
        # Get completed subtasks
        subtasks = detect_subtasks(episode)
        
        # If no subtasks completed, return default task ID
        if not subtasks:
            return 0
        
        # Determine task ID based on combination of completed subtasks
        subtask_str = "_".join(sorted(subtasks))
        
        # Map to task ID using hash function (0-4)
        task_id = hash(subtask_str) % 5
        return task_id
    
    except Exception as e:
        print(f"Error determining task ID: {e}")
        return 0  # Return default task ID


def process_episode(episode, max_len=MAX_LEN):
    """Process a single episode into training sequences"""
    obs = torch.tensor(episode.observations["observation"][:-1], dtype=torch.float32)
    acts = torch.tensor(episode.actions, dtype=torch.float32)
    rews = torch.tensor(episode.rewards, dtype=torch.float32)
    rtg = rews.flip([0]).cumsum(0).flip([0]).unsqueeze(-1)
    prev_acts = torch.cat([torch.zeros_like(acts[:1]), acts[:-1]], dim=0)
    timesteps = torch.arange(len(obs)).unsqueeze(-1)
    
    sequences = []
    if obs.shape[0] < max_len: 
        return sequences
        
    for i in range(obs.shape[0] - max_len + 1):
        sequences.append({
            "observations": obs[i:i+max_len],
            "actions": acts[i:i+max_len],
            "reward": rews[i:i+max_len].unsqueeze(-1),
            "return_to_go": rtg[i:i+max_len],
            "prev_actions": prev_acts[i:i+max_len],
            "timesteps": timesteps[i:i+max_len],
        })
    return sequences


def organize_data_by_task(dataset, max_len=MAX_LEN):
    """
    Organize dataset by task ID using functions 
    to see what subtask is completed within each trajectory
    """
    
    print("Organizing data by task...")
    task_datasets = defaultdict(list)
    task_counts = defaultdict(int)
    
    for episode in tqdm(dataset):
        try:
            task_id = determine_task_id(episode)
            task_counts[task_id] += 1
            
            # Process episode and add sequences to task dataset
            sequences = process_episode(episode, max_len)
            task_datasets[task_id].extend(sequences)
        except Exception as e:
            print(f"Error processing episode: {e}")
            continue
    
    for task_id, data in task_datasets.items():
        print(f"Task {task_id}: {len(data)} sequences, {task_counts[task_id]} episodes")
    
    return task_datasets


def get_batches(data, batch_size=BATCH_SIZE):
    """Generate batches from task data"""
    if len(data) < batch_size:
        # If data is smaller than batch_size, sample with replacement
        indices = np.random.choice(len(data), size=batch_size, replace=True)
    else:
        # Otherwise, shuffle indices
        indices = np.random.permutation(len(data))
        
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        if len(batch_indices) < batch_size:
            # Fill remaining slots in batch by sampling with replacement
            extra = np.random.choice(
                len(data), 
                size=batch_size-len(batch_indices), 
                replace=True
            )
            batch_indices = np.concatenate([batch_indices, extra])
            
        # batch by stacking examples
        batch = [data[idx % len(data)] for idx in batch_indices]
        yield {k: torch.stack([d[k] for d in batch]).to(device) for k in batch[0]}


def update_parameters(optimizer, loss, params):
    """Update parameters with loss gradient"""
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    
    # clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()


def moving_average(data, window_size=WINDOW_SIZE):
    """Calculate moving average over a window"""
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def exponential_moving_average(data, alpha=SMOOTH_ALPHA):
    """Calculate exponential moving average with smoothing factor alpha"""
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def main():
    """Main training function implementing Meta-Planning as Inference"""
   
    print("Loading dataset...")
    dataset = minari.load_dataset("D4RL/kitchen/mixed-v2", download=True)
    
    task_datasets = organize_data_by_task(dataset)
    
    print("Initializing model...")
    state_dim = dataset[0].observations["observation"].shape[1]
    act_dim = dataset[0].actions.shape[1]
    model = MPILearningLearner(
        state_dim=state_dim, 
        act_dim=act_dim, 
        context_len=MAX_LEN, 
        h_dim=HIDDEN_SIZE,
        device=device
    ).to(device)
    
    beta_params = list(model.trajectory_generator.parameters())
    gamma_params = list(model.reward_head.parameters())
    alpha_params = list(model.generator.parameters())
    phi_params = list(model.ll_encoder.parameters())
    psi_params = list(model.ll_decoder.parameters())
    
    beta_optimizer = torch.optim.Adam(beta_params, lr=LR_TRAJ_GEN)
    gamma_optimizer = torch.optim.Adam(gamma_params, lr=LR_REWARD)
    alpha_optimizer = torch.optim.Adam(alpha_params, lr=LR)
    ll_optimizer = torch.optim.Adam(phi_params + psi_params, lr=LR_ENCODER)
    
    beta_scheduler = StepLR(beta_optimizer, step_size=2, gamma=0.9)
    gamma_scheduler = StepLR(gamma_optimizer, step_size=2, gamma=0.9)
    alpha_scheduler = StepLR(alpha_optimizer, step_size=2, gamma=0.9)
    ll_scheduler = StepLR(ll_optimizer, step_size=2, gamma=0.9)
    
    losses = []
    action_losses = []
    reward_losses = []
    alpha_losses = []
    kl_losses = []
    task_alpha_values = {}
    alpha_prime = None
    
    running_loss = None
    running_r_loss = None
    running_a_loss = None
    running_alpha_loss = None
    running_kl_loss = None

    # tracking per-task performance
    task_losses = {task_id: [] for task_id in task_datasets.keys()}
    
    print("Starting training...")
    for epoch in range(N_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{N_EPOCHS} =====")
        epoch_losses = []
        epoch_action_losses = []
        epoch_reward_losses = []
        epoch_alpha_losses = []
        epoch_kl_losses = []
        epoch_task_losses = defaultdict(list)
        
        for task_id, task_data in task_datasets.items():
            print(f"Processing task {task_id}...")
            task_alphas = []
            task_epoch_losses = []
            
            n_batches = min(5, len(task_data) // BATCH_SIZE + 1)
            
            for batch_idx, batch in enumerate(get_batches(task_data)):
                if batch_idx >= n_batches:
                    break
                    
                pred_action, pred_reward, alpha_k, alpha_loss, kl, mu_alpha, _ = model(
                    batch["observations"], 
                    batch["prev_actions"], 
                    batch["reward"], 
                    batch["timesteps"].squeeze(-1), 
                    alpha_bar=alpha_prime # give alpha prime from previous training for supervision
                )
                
                loss_r = F.mse_loss(pred_reward, batch["reward"][:, -1, 0])
                loss_a = F.mse_loss(pred_action, batch["actions"][:, -1])
                total_loss = loss_a + loss_r + KL_WEIGHT * kl + ALPHA_WEIGHT * alpha_loss
                
                model.zero_grad()
                
                # trajectory generator parameters β
                update_parameters(beta_optimizer, loss_a, beta_params)
                
                # reward model parameters γ
                update_parameters(gamma_optimizer, loss_r, gamma_params)
                
                # fast learning parameters α
                alpha_loss_term = KL_WEIGHT * kl + ALPHA_WEIGHT * alpha_loss
                update_parameters(alpha_optimizer, alpha_loss_term, alpha_params)
                
                # LL parameters (φ and ψ)
                ll_loss = KL_WEIGHT * kl + ALPHA_WEIGHT * alpha_loss
                update_parameters(ll_optimizer, ll_loss, phi_params + psi_params)
                
                # mu_alpha for this task, we do weak supervision on mu_alpha
                batch_averaged_mu_alpha = mu_alpha.mean(dim=0) # [B, alpha_dim] -> [alpha_dim]
                task_alphas.append(batch_averaged_mu_alpha.detach())
                
                loss_val = total_loss.item()
                r_loss_val = loss_r.item()
                a_loss_val = loss_a.item()
                alpha_loss_val = alpha_loss.item()
                kl_loss_val = kl.item()
            
                if running_loss is None:
                    running_loss = loss_val
                    running_r_loss = r_loss_val
                    running_a_loss = a_loss_val
                    running_alpha_loss = alpha_loss_val
                    running_kl_loss = kl_loss_val
                else:
                    running_loss = 0.9 * running_loss + 0.1 * loss_val
                    running_r_loss = 0.9 * running_r_loss + 0.1 * r_loss_val
                    running_a_loss = 0.9 * running_a_loss + 0.1 * a_loss_val
                    running_alpha_loss = 0.9 * running_alpha_loss + 0.1 * alpha_loss_val
                    running_kl_loss = 0.9 * running_kl_loss + 0.1 * kl_loss_val
                
                epoch_losses.append(loss_val)
                epoch_action_losses.append(a_loss_val)
                epoch_reward_losses.append(r_loss_val)
                epoch_alpha_losses.append(alpha_loss_val)
                epoch_kl_losses.append(kl_loss_val)
                task_epoch_losses.append(loss_val)
                
                print(f"  Batch {batch_idx+1}/{n_batches}, Loss: {loss_val:.4f}, Running Avg: {running_loss:.4f}, "
                      f"Aaction-Loss: {a_loss_val:.4f}, Reward-Loss: {r_loss_val:.4f}, Alpha-Loss: {alpha_loss_val:.6f}")
            
            if task_alphas:
                task_alpha_values[task_id] = torch.stack(task_alphas).mean(dim=0)  # [alpha_dim]
                print(f"Task {task_id} alpha shape after averaging: {task_alpha_values[task_id].shape}")
            
            epoch_task_losses[task_id] = task_epoch_losses
            task_losses[task_id].extend(task_epoch_losses)
        
        # Update α'
        if task_alpha_values:
            try:
                # stack alpha values from all tasks
                all_alphas = torch.stack(list(task_alpha_values.values()))  # [num_tasks, alpha_dim]
                print(f"all_alphas shape: {all_alphas.shape}")
                
                # weighted average across tasks
                weights = torch.ones(len(all_alphas), device=device) / len(all_alphas)
                new_alpha_prime = torch.sum(all_alphas * weights.unsqueeze(-1), dim=0)  # [alpha_dim]
                
                # slow learning update of α'
                if alpha_prime is None:
                    alpha_prime = new_alpha_prime
                else:
                    alpha_prime = ALPHA_P_MOMENTUM * alpha_prime + (1 - ALPHA_P_MOMENTUM) * new_alpha_prime
                
                print(f"Updated alpha' norm: {alpha_prime.norm().item():.4f}, shape: {alpha_prime.shape}")
            
            except Exception as e:
                print(f"Error calculating alpha': {e}")
                if alpha_prime is None:
                    alpha_prime = torch.zeros(HIDDEN_SIZE, device=device)
        
        beta_scheduler.step()
        gamma_scheduler.step()
        alpha_scheduler.step()
        ll_scheduler.step()
        
        losses.extend(epoch_losses)
        action_losses.extend(epoch_action_losses)
        reward_losses.extend(epoch_reward_losses)
        alpha_losses.extend(epoch_alpha_losses)
        kl_losses.extend(epoch_kl_losses)
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_action_loss = np.mean(epoch_action_losses) if epoch_action_losses else 0
        avg_reward_loss = np.mean(epoch_reward_losses) if epoch_reward_losses else 0
        avg_alpha_loss = np.mean(epoch_alpha_losses) if epoch_alpha_losses else 0
        
        print("\nTask-specific performance:")
        for task_id, task_losses_epoch in epoch_task_losses.items():
            if task_losses_epoch:
                task_avg = np.mean(task_losses_epoch)
                print(f"  Task {task_id}: Avg Loss = {task_avg:.4f}")
        
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Action loss: {avg_action_loss:.4f}")
        print(f"  Reward loss: {avg_reward_loss:.4f}")
        print(f"  Alpha loss: {avg_alpha_loss:.6f}")
        print(f"  Running average loss: {running_loss:.4f}")
    
    print("\nMPI-LL training complete. Generating visualizations...")
    
    plt.figure(figsize=(18, 12))
    
    # Raw losses
    plt.subplot(2, 3, 1)
    plt.plot(losses, alpha=0.3, label="Total loss (raw)")
    plt.plot(action_losses, alpha=0.3, label="Action loss (raw)")
    plt.plot(reward_losses, alpha=0.3, label="Reward loss (raw)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Raw Training Losses")
    plt.legend()
    plt.grid(True)
    
    # Moving average losses
    plt.subplot(2, 3, 2)
    window = min(WINDOW_SIZE, len(losses))
    plt.plot(moving_average(losses, window), label="Total loss (MA)")
    plt.plot(moving_average(action_losses, window), label="Action loss (MA)")
    plt.plot(moving_average(reward_losses, window), label="Reward loss (MA)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Moving Average (window={window})")
    plt.legend()
    plt.grid(True)
    
    # Exponential moving average losses
    plt.subplot(2, 3, 3)
    plt.plot(exponential_moving_average(losses, SMOOTH_ALPHA), label="Total loss (EMA)")
    plt.plot(exponential_moving_average(action_losses, SMOOTH_ALPHA), label="Action loss (EMA)")
    plt.plot(exponential_moving_average(reward_losses, SMOOTH_ALPHA), label="Reward loss (EMA)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Exponential Moving Average (alpha={SMOOTH_ALPHA})")
    plt.legend()
    plt.grid(True)
    
    # Per-task losses
    plt.subplot(2, 3, 4)
    for task_id, task_loss in task_losses.items():
        if task_loss:  # Skip empty lists
            plt.plot(exponential_moving_average(task_loss, SMOOTH_ALPHA), 
                    label=f"Task {task_id}")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Per-Task Loss (EMA)")
    plt.legend()
    plt.grid(True)
    
    # Alpha and KL losses
    plt.subplot(2, 3, 5)
    plt.plot(exponential_moving_average(alpha_losses, SMOOTH_ALPHA), label="Alpha Loss")
    plt.plot(exponential_moving_average(kl_losses, SMOOTH_ALPHA), label="KL Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Alpha & KL Losses (EMA)")
    plt.legend()
    plt.grid(True)
    
    # Loss ratio (Action vs Reward)
    plt.subplot(2, 3, 6)
    # Calculate ratio of action loss to total loss
    ratio = np.array(action_losses) / (np.array(action_losses) + np.array(reward_losses) + 1e-8)
    plt.plot(exponential_moving_average(ratio, SMOOTH_ALPHA), label="Action/(Action+Reward) Ratio")
    plt.axhline(y=0.5, color='r', linestyle='--', label="Equal contribution")
    plt.xlabel("Training Step")
    plt.ylabel("Ratio")
    plt.title("Action vs Reward Loss Contribution")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True)
    
    plt.tight_layout()
    # plt.savefig("mpill_comprehensive_analysis.png", dpi=300)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    
    # task performance comparison
    task_avg_losses = {}
    for task_id, task_loss in task_losses.items():
        if task_loss:
            # Split into first and second half to show improvement
            half_point = len(task_loss) // 2
            first_half = task_loss[:half_point]
            second_half = task_loss[half_point:]
            
            task_avg_losses[task_id] = {
                'overall': np.mean(task_loss),
                'first_half': np.mean(first_half) if first_half else 0,
                'second_half': np.mean(second_half) if second_half else 0
            }
    
    # comparative performance
    task_ids = list(task_avg_losses.keys())
    if task_ids:
        overall_avgs = [task_avg_losses[tid]['overall'] for tid in task_ids]
        first_half_avgs = [task_avg_losses[tid]['first_half'] for tid in task_ids]
        second_half_avgs = [task_avg_losses[tid]['second_half'] for tid in task_ids]
        
        x = np.arange(len(task_ids))
        width = 0.25
        
        plt.bar(x - width, first_half_avgs, width, label='First Half')
        plt.bar(x, overall_avgs, width, label='Overall')
        plt.bar(x + width, second_half_avgs, width, label='Second Half')
        
        plt.xlabel('Task ID')
        plt.ylabel('Average Loss')
        plt.title('Task Performance Comparison')
        plt.xticks(x, task_ids)
        plt.legend()
        plt.grid(True, axis='y')
        
        # plt.savefig("mpill_task_comparison.png", dpi=300)
        plt.show()
    
    # torch.save(model.state_dict(), "mpill_model.pt")
    # print("Model saved and analysis complete.")

if __name__ == "__main__":
    main()