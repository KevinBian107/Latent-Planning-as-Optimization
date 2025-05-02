import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import minari
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.LPT import LatentPlannerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

device = torch.device("cpu")

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())

MAX_LEN = 15
HIDDEN_SIZE = 32
N_LAYER = 3
N_HEAD = 1
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

WINDOW_SIZE = 30  # Window size for running average
SMOOTH_ALPHA = 0.05  # Exponential moving average smoothing factor

context_len = MAX_LEN

MICROWAVE_IDX = 31    # Microwave door position index
KETTLE_IDX_X = 32     # Kettle x-coordinate index
KETTLE_IDX_Y = 33     # Kettle y-coordinate index
KETTLE_IDX_Z = 34     # Kettle z-coordinate index
LIGHT_SWITCH_IDX = 26 # Light switch position index
SLIDE_CABINET_IDX = 28 # Sliding cabinet door position index

MICROWAVE_THRESHOLD = 0.2      # Threshold for open microwave
KETTLE_MOVE_THRESHOLD = 0.1    # Threshold for moved kettle
LIGHT_SWITCH_THRESHOLD = -0.6  # Threshold for on light switch
SLIDE_CABINET_THRESHOLD = 0.2  # Threshold for open sliding cabinet

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
    
    # Get initial and final states
    initial_state = observations[0]
    final_state = observations[-1]
    
    # Check for completed subtasks
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
    """Determine task ID based on completed subtasks"""
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
    observations = torch.tensor(episode.observations["observation"][:-1], dtype=torch.float32).to(device)
    actions = torch.tensor(episode.actions, dtype=torch.float32).to(device)
    rew = torch.tensor(episode.rewards, dtype=torch.float32).to(device)
    done = torch.tensor(episode.terminations, dtype=torch.bool).to(device)

    rtg = rew.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
    prev_act = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
    timesteps = torch.arange(len(observations), dtype=torch.long, device=device).unsqueeze(-1)

    sequences = []
    if observations.shape[0] < max_len:
        return sequences

    for i in range(observations.shape[0] - max_len + 1):
        sequences.append({
            "observations": observations[i:i+max_len],
            "actions": actions[i:i+max_len],
            "reward": rew[i:i+max_len].unsqueeze(-1),
            "done": done[i:i+max_len].unsqueeze(-1),
            "return_to_go": rtg[i:i+max_len],
            "prev_actions": prev_act[i:i+max_len],
            "timesteps": timesteps[i:i+max_len],
        })
    return sequences


def organize_data_by_task(dataset, max_len=MAX_LEN):
    """Organize dataset by task ID"""
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
            
        # Create batch by stacking examples
        batch = [data[idx % len(data)] for idx in batch_indices]
        yield {k: torch.stack([d[k] for d in batch]).to(device) for k in batch[0]}

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
    """Main training function implementing LPT"""
    print("Loading dataset...")
    dataset = minari.load_dataset('D4RL/kitchen/mixed-v2', download=True)

    # Organize data by task
    task_datasets = organize_data_by_task(dataset)

    state_dim = dataset[0].observations["observation"].shape[1]
    act_dim = dataset[0].actions.shape[1]

    model = LatentPlannerModel(
        state_dim=state_dim,
        act_dim=act_dim,
        h_dim=HIDDEN_SIZE,
        context_len=context_len,
        n_blocks=N_LAYER,
        n_heads=N_HEAD,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    all_losses = []
    r_losses = []
    a_losses = []

    running_loss = None
    running_r_loss = None
    running_a_loss = None

    # For tracking per-task performance
    task_losses = {task_id: [] for task_id in task_datasets.keys()}

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        epoch_losses = []
        epoch_r_losses = []
        epoch_a_losses = []
        epoch_task_losses = defaultdict(list)
        
        # Process each task separately
        for task_id, task_data in task_datasets.items():
            print(f"Processing task {task_id}...")
            task_epoch_losses = []
            
            n_batches = min(5, len(task_data) // BATCH_SIZE + 1)
            
            for batch_idx, batch in enumerate(get_batches(task_data)):
                if batch_idx >= n_batches:
                    break
                    
                batch_inds = torch.arange(BATCH_SIZE, device=device)
                
                pred_action, pred_state, pred_reward = model(
                    states=batch["observations"],
                    actions=batch["prev_actions"],
                    timesteps=batch["timesteps"].squeeze(-1),
                    rewards=batch["reward"],
                    batch_inds=batch_inds,
                )
                
                loss_r = torch.nn.MSELoss()(pred_reward, batch["reward"][:, -1, 0])
                loss_a = torch.nn.MSELoss()(pred_action, batch["actions"][:, -1])
                
                total_loss = loss_r + loss_a
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                loss_val = total_loss.item()
                r_loss_val = loss_r.item()
                a_loss_val = loss_a.item()
                
                if running_loss is None:
                    running_loss = loss_val
                    running_r_loss = r_loss_val
                    running_a_loss = a_loss_val
                else:
                    running_loss = 0.9 * running_loss + 0.1 * loss_val
                    running_r_loss = 0.9 * running_r_loss + 0.1 * r_loss_val
                    running_a_loss = 0.9 * running_a_loss + 0.1 * a_loss_val
                
                epoch_losses.append(loss_val)
                epoch_r_losses.append(r_loss_val)
                epoch_a_losses.append(a_loss_val)
                task_epoch_losses.append(loss_val)
                
                print(f"  Batch {batch_idx+1}/{n_batches}, Loss: {loss_val:.4f}, Running Avg: {running_loss:.4f}, "
                    f"Action-Loss: {a_loss_val:.4f}, Reward-Loss: {r_loss_val:.4f}")
            
            epoch_task_losses[task_id] = task_epoch_losses
            task_losses[task_id].extend(task_epoch_losses)
            
        all_losses.extend(epoch_losses)
        r_losses.extend(epoch_r_losses)
        a_losses.extend(epoch_a_losses)
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_r_loss = np.mean(epoch_r_losses) if epoch_r_losses else 0
        avg_a_loss = np.mean(epoch_a_losses) if epoch_a_losses else 0
        
        print("\nTask-specific performance:")
        for task_id, losses in epoch_task_losses.items():
            if losses:
                task_avg = np.mean(losses)
                print(f"  Task {task_id}: Avg Loss = {task_avg:.4f}")
        
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Reward loss: {avg_r_loss:.4f}")
        print(f"  Action loss: {avg_a_loss:.4f}")
        print(f"  Running average loss: {running_loss:.4f}")

    print("\nLPT training complete. Generating visualizations...")

    plt.figure(figsize=(18, 12))

    # Raw losses
    plt.subplot(2, 3, 1)
    plt.plot(all_losses, alpha=0.3, label="Total loss (raw)")
    plt.plot(a_losses, alpha=0.3, label="Action loss (raw)")
    plt.plot(r_losses, alpha=0.3, label="Reward loss (raw)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Raw Training Losses")
    plt.legend()
    plt.grid(True)

    # Moving average losses
    plt.subplot(2, 3, 2)
    window = min(WINDOW_SIZE, len(all_losses))
    plt.plot(moving_average(all_losses, window), label="Total loss (MA)")
    plt.plot(moving_average(a_losses, window), label="Action loss (MA)")
    plt.plot(moving_average(r_losses, window), label="Reward loss (MA)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Moving Average (window={window})")
    plt.legend()
    plt.grid(True)

    # Exponential moving average losses
    plt.subplot(2, 3, 3)
    plt.plot(exponential_moving_average(all_losses, SMOOTH_ALPHA), label="Total loss (EMA)")
    plt.plot(exponential_moving_average(a_losses, SMOOTH_ALPHA), label="Action loss (EMA)")
    plt.plot(exponential_moving_average(r_losses, SMOOTH_ALPHA), label="Reward loss (EMA)")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title(f"Exponential Moving Average (α={SMOOTH_ALPHA})")
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

    # Loss distribution
    plt.subplot(2, 3, 5)
    plt.hist(all_losses, bins=30, alpha=0.7, label="Total")
    plt.hist(a_losses, bins=30, alpha=0.5, label="Action")
    plt.hist(r_losses, bins=30, alpha=0.5, label="Reward")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.title("Loss Distribution")
    plt.legend()
    plt.grid(True)

    # Loss ratio (Action vs Reward)
    plt.subplot(2, 3, 6)
    # Calculate ratio of action loss to total loss
    ratio = np.array(a_losses) / (np.array(all_losses) + 1e-8)
    plt.plot(exponential_moving_average(ratio, SMOOTH_ALPHA), label="Action/Total Ratio")
    plt.axhline(y=0.5, color='r', linestyle='--', label="Equal contribution")
    plt.xlabel("Training Step")
    plt.ylabel("Ratio")
    plt.title("Action Loss Contribution")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True)

    plt.tight_layout()
    # plt.savefig("lpt_comprehensive_analysis.png", dpi=300)
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
        
        # plt.savefig("lpt_task_comparison.png", dpi=300)
        plt.show()

    # torch.save(model.state_dict(), "lpt_task_sequential_model.pt")
    # print("Model saved and analysis complete.")

if __name__ == "__main__":
    main()