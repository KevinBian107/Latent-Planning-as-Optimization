import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import minari
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import pdb

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.MPILL import MPILearningLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")

device = torch.device("cpu")

MAX_LEN = 20 # Horizon length
HIDDEN_SIZE = 512
BATCH_SIZE = 16
N_EPOCHS = 20
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

def segment_trajectory_by_subtasks(
    full_episode, 
    task_goal_keys, 
    proximity_thresholds,
    stability_duration
):
    """
    Segments a single long trajectory into multiple sub-trajectories based on
    stable completion of sub-tasks. A sub-task is complete when its achieved_goal 
    stays close to its desired_goal for a specified stability_duration.

    Args:
        full_episode: A dictionary-like object representing one full episode.
                      - full_episode.observations['achieved_goal'][task_key] is a sequence (step, D_task_goal_dim)
                      - full_episode.observations['desired_goal'][task_key] is a static target (step, D_task_goal_dim)
        task_goal_keys: List of strings for sub-task keys.
        proximity_thresholds: Dict mapping task_key to a proximity threshold.
        stability_duration: Integer, number of timesteps for stability.

    Returns:
        A list of dictionaries, where each dictionary is a segmented sub-trajectory.
        Each sub-trajectory will have an 'task_id' field.
    """
    segmented_trajectories = []

    all_obs_data = full_episode.observations
    all_actions = full_episode.actions
    all_rewards = full_episode.rewards
    all_terminations = full_episode.terminations
    all_truncations = full_episode.truncations

    num_total_steps = len(all_actions)
    if num_total_steps == 0 or stability_duration <= 0:
        return []

    current_segment_start_idx = 0
    
    # Tracks consecutive timesteps each task has been "close"
    task_close_streaks = {key: 0 for key in task_goal_keys}
    
    # Tracks which of the main tasks have been segmented
    tasks_segmented_this_episode = set()

    for t in range(num_total_steps):
        task_to_segment = None

        for task_key in task_goal_keys:
            if task_key in tasks_segmented_this_episode:
                continue # This task has already been segmented in this episode

            if task_key not in all_obs_data.get('achieved_goal', {}) or \
                task_key not in all_obs_data.get('desired_goal', {}):
                raise ValueError(f"Task key '{task_key}' not found in observations.")
                # if task_key in task_close_streaks: del task_close_streaks[task_key] # Stop tracking
                # continue

            current_achieved_state_for_task = all_obs_data['achieved_goal'][task_key][t]
            desired_state_for_task = all_obs_data['desired_goal'][task_key][0]

            # Calculate difference
            # Vector goals
            if isinstance(current_achieved_state_for_task, np.ndarray) and current_achieved_state_for_task.ndim > 0:
                diff = np.linalg.norm(current_achieved_state_for_task - desired_state_for_task)
            else: # Scalar goals
                diff = np.abs(current_achieved_state_for_task - desired_state_for_task)

            prox_threshold = proximity_thresholds.get(task_key)
            if prox_threshold is None:
                raise ValueError(f"Proximity threshold for task '{task_key}' not provided.")
                # if task_key in task_close_streaks: del task_close_streaks[task_key]
                # continue

            if diff < prox_threshold:
                task_close_streaks[task_key] += 1
            else:
                task_close_streaks[task_key] = 0

            # Check if this task met the stability duration
            if task_close_streaks[task_key] >= stability_duration:
                task_to_segment = task_key
                break # Prioritize this task for segmentation at this timestep
        
        # ---starts segmenting when a task is stable--- #
        if task_to_segment:
            segment_end_idx = t 
            # print(f"Task '{task_to_segment}' detected as stable and completed at timestep {segment_end_idx}.")

            segment = {}
            segment_observations = {}

            # Copy the segment of the trajectory
            achieved_goal = all_obs_data['achieved_goal'][task_to_segment][current_segment_start_idx : segment_end_idx + 1]
            desired_goal = all_obs_data['desired_goal'][task_to_segment][current_segment_start_idx : segment_end_idx + 1]
            observation = all_obs_data['observation'][current_segment_start_idx : segment_end_idx + 1]

            segment_observations['achieved_goal'] = {}
            segment_observations['achieved_goal'][task_to_segment] = achieved_goal
            segment_observations['desired_goal'] = {}
            segment_observations['desired_goal'][task_to_segment] = desired_goal

            segment_observations['observation'] = observation

            segment['observations'] = segment_observations
            segment['actions'] = all_actions[current_segment_start_idx : segment_end_idx + 1]
            segment['rewards'] = all_rewards[current_segment_start_idx : segment_end_idx + 1]

            segment_terminations = np.zeros_like(segment['rewards'], dtype=bool)
            if len(segment_terminations) > 0:
                segment_terminations[-1] = True
            segment['terminations'] = segment_terminations
            segment['truncations'] = np.zeros_like(segment['rewards'], dtype=bool)

            segment['task_id'] = task_to_segment
            segmented_trajectories.append(segment)

            tasks_segmented_this_episode.add(task_to_segment)
            current_segment_start_idx = segment_end_idx + 1
            # Reset all streaks as a new task/segment begins
            task_close_streaks = {key: 0 for key in task_goal_keys if key not in tasks_segmented_this_episode}

            if not task_close_streaks: 
                break
    
    remaining_task = list(set(task_goal_keys) - tasks_segmented_this_episode)
    if len(remaining_task) > 0:
        remaining_task = remaining_task[0]
    else:
        remaining_task = None

    # Handle any remaining part of the trajectory
    if current_segment_start_idx < num_total_steps and len(tasks_segmented_this_episode) < len(task_goal_keys):
        # print(f"Adding trailing segment from timestep {current_segment_start_idx} to {num_total_steps -1}.")
        trailing_segment = {}
        trailing_segment_observations = {}

        # Copy the segment of the trajectory
        achieved_goal = all_obs_data['achieved_goal'][remaining_task][current_segment_start_idx:]
        desired_goal = all_obs_data['desired_goal'][remaining_task][current_segment_start_idx:]
        observation = all_obs_data['observation'][current_segment_start_idx:]

        trailing_segment_observations['achieved_goal'] = {}
        trailing_segment_observations['achieved_goal'][remaining_task] = achieved_goal
        trailing_segment_observations['desired_goal'] = {}
        trailing_segment_observations['desired_goal'][remaining_task] = desired_goal

        trailing_segment_observations['observation'] = observation

        trailing_segment['observations'] = trailing_segment_observations
        trailing_segment['actions'] = all_actions[current_segment_start_idx:]
        trailing_segment['rewards'] = all_rewards[current_segment_start_idx:]
        trailing_segment['terminations'] = all_terminations[current_segment_start_idx:]
        trailing_segment['truncations'] = all_truncations[current_segment_start_idx:]
        trailing_segment['task_id'] = remaining_task
        segmented_trajectories.append(trailing_segment)

    return segmented_trajectories

TASK_KEYS = ['microwave', 'kettle', 'light switch', 'slide cabinet'] 
PROXIMITY_THRESHOLDS = {
    'microwave': 0.2,       
    'kettle': 0.3,         
    'light switch': 0.2,    
    'slide cabinet': 0.2   
}
STABILITY_DURATION = 20

def split_task(dataset):
    """
    Splits the dataset into sub-tasks based on the specified keys and proximity thresholds.
    
    Args:
        dataset: A list of trajectories to be segmented.
    
    Returns:
        segmented_dataset layout:
        { 
            'microwave': [traj, ...],  
            'kettle': [traj, ...], 
            ... 
        }

        Within a traj, the layout looks like the following:
        { 
        'task_id': ... 
        'observations' : { 
            'achieved_goal': {'task_name': ...} 
            'desired_goal': {'task_name': ...} 
            'observation': ... 
        } 
        'actions': ... 
        'rewards': ... 
        'terminations': ... 
        'truncations': ... 
        }
    """

    segmented_dataset = defaultdict(list)
    task_counts = defaultdict(int)

    for i, traj in enumerate(dataset):
        segmented_traj = segment_trajectory_by_subtasks(
            traj, 
            TASK_KEYS,
            PROXIMITY_THRESHOLDS,
            STABILITY_DURATION
        )

        for segment in segmented_traj:
            task_id = segment['task_id']
            
            # cut segment to equal lenght of sliding window 
            sequences = process_episode(segment)
            segmented_dataset[task_id].extend(sequences)

            task_counts[task_id] += len(sequences)
        
        print('processed trajectory', i, 'of', len(dataset))
    
    for task_id, num_sequence in task_counts.items():
        print(f"Task {task_id}: {num_sequence} sequences")

    return segmented_dataset


def process_episode(episode, max_len=MAX_LEN):
    """Process a single episode into training sequences"""
    obs = torch.tensor(episode['observations']["observation"][:-1], dtype=torch.float32)
    acts = torch.tensor(episode['actions'], dtype=torch.float32)
    rews = torch.tensor(episode['rewards'], dtype=torch.float32)
    rtg = rews.flip([0]).cumsum(0).flip([0]).unsqueeze(-1)
    prev_acts = torch.cat([torch.zeros_like(acts[:1]), acts[:-1]], dim=0)
    timesteps = torch.arange(len(obs)).unsqueeze(-1)
    
    sequences = []
    # if the episode is shorter than max_len, add it as a single sequence
    if obs.shape[0] < max_len: 
        # pad the sequence to max_len
        pad_len = max_len - obs.shape[0]
        obs = torch.cat([obs, torch.zeros(max_len - obs.shape[0], obs.shape[1], dtype=obs.dtype)], dim=0)
        acts = torch.cat([acts, torch.zeros(max_len - acts.shape[0], acts.shape[1], dtype=acts.dtype)], dim=0)
        rews = torch.cat([rews, torch.zeros(max_len - rews.shape[0], dtype=rews.dtype)], dim=0)
        rtg = torch.cat([rtg, torch.zeros(max_len - rtg.shape[0], 1, dtype=rtg.dtype)], dim=0)
        prev_acts = torch.cat([prev_acts, torch.zeros(max_len - prev_acts.shape[0], prev_acts.shape[1], dtype=prev_acts.dtype)], dim=0)
        timesteps = torch.cat([timesteps, torch.zeros(max_len - timesteps.shape[0], 1, dtype=timesteps.dtype)], dim=0)

        # add the sequence to the list
        sequences.append({
            "observations": obs, 
            "actions": acts,
            "reward": rews.unsqueeze(-1), 
            "return_to_go": rtg, 
            "prev_actions": prev_acts,
            "timesteps": timesteps,
        })
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
    # dataset = minari.load_dataset("D4RL/kitchen/mixed-v2", download=True)
    dataset = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
    task_datasets = split_task(dataset)

    print("Initializing model...")
    state_dim = dataset[0].observations["observation"].shape[1]
    act_dim   = dataset[0].actions.shape[1]
    model = MPILearningLearner(
        state_dim=state_dim,
        act_dim=act_dim,
        context_len=MAX_LEN,
        h_dim=HIDDEN_SIZE,
        device=device
    ).to(device)

    beta_params  = list(model.trajectory_generator.parameters())
    gamma_params = list(model.reward_head.parameters())
    alpha_params = list(model.generator.parameters())
    phi_params   = list(model.ll_encoder.parameters())
    psi_params   = list(model.ll_decoder.parameters())

    beta_optimizer  = torch.optim.Adam(beta_params,  lr=LR_TRAJ_GEN)
    gamma_optimizer = torch.optim.Adam(gamma_params, lr=LR_REWARD)
    alpha_optimizer = torch.optim.Adam(alpha_params, lr=LR)
    ll_optimizer    = torch.optim.Adam(phi_params + psi_params, lr=LR_ENCODER)

    beta_scheduler  = StepLR(beta_optimizer,  step_size=2, gamma=0.9)
    gamma_scheduler = StepLR(gamma_optimizer, step_size=2, gamma=0.9)
    alpha_scheduler = StepLR(alpha_optimizer, step_size=2, gamma=0.9)
    ll_scheduler    = StepLR(ll_optimizer,    step_size=2, gamma=0.9)

    # Global logs for plotting
    losses        = []
    action_losses = []
    reward_losses = []
    alpha_losses  = []
    kl_losses     = []
    alpha_prime   = None

    # Running averages
    running_loss        = running_r_loss = running_a_loss = None
    running_alpha_loss  = running_kl_loss = None

    # Per-task histories
    task_losses = {tid: [] for tid in task_datasets.keys()}

    # Helper to limit number of batches per task
    def limited_batches(data, max_batches):
        for idx, batch in enumerate(get_batches(data)):
            if idx >= max_batches:
                break
            yield batch

    # Pre-compute how many batches per task
    n_batches_per_task = {
        tid: min(5, len(td) // BATCH_SIZE + 1)
        for tid, td in task_datasets.items()
    }
    task_ids = list(task_datasets.keys())

    print("Starting training...")
    for epoch in range(N_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{N_EPOCHS} =====")
        # Per-epoch buffers
        epoch_losses        = []
        epoch_action_losses = []
        epoch_reward_losses = []
        epoch_alpha_losses  = []
        epoch_kl_losses     = []
        epoch_task_losses   = defaultdict(list)

        # Re-create batch iterators for each task
        # task_datasets[tid] is a list with 962 items (batch) for microwave, each item is a dict, each with length of something (i.e. 20 time steps)
        # In total 962 个batch x 20 个time steps每个batch
        # during learning, we are doing multi-step leraning in one pass, just like LPT
        
        batch_iters = [
            limited_batches(task_datasets[tid], n_batches_per_task[tid])
            for tid in task_ids
        ]

        # Outer: one batch from each task to constitute one batch_group
        # batch_iters is generating 4 tasks, batch_group is a tupleof length 4
        for batch_group in zip(*batch_iters):
            task_alpha_vals = {}

            # Inner: process each task in this batch_group
            for tid, task_batch in zip(task_ids, batch_group):
                # batch is a dict with ['observations', 'actions', 'reward', 'return_to_go', 'prev_actions', 'timesteps']
                
                # print('TEST', alpha_prime) should be same for each task in one batch_group
                
                # 16 batch, 20 timesetp, and 59 observations
                pred_action, pred_reward, alpha_k, alpha_loss, kl, mu_alpha, _ = model(
                    task_batch["observations"],
                    task_batch["prev_actions"],
                    task_batch["reward"],
                    task_batch["timesteps"].squeeze(-1),
                    alpha_bar=alpha_prime
                )

                loss_r    = F.mse_loss(pred_reward, task_batch["reward"][:, -1, 0])
                loss_a    = F.mse_loss(pred_action,  task_batch["actions"][:, -1])
                fast_loss = KL_WEIGHT * kl + ALPHA_WEIGHT * alpha_loss
                total_loss = loss_a + loss_r + fast_loss

                model.zero_grad()
                update_parameters(beta_optimizer,  loss_a,    beta_params)
                update_parameters(gamma_optimizer, loss_r,    gamma_params)
                update_parameters(alpha_optimizer, fast_loss, alpha_params)
                update_parameters(ll_optimizer,    fast_loss, phi_params + psi_params) # appending

                # Metrics
                lv     = total_loss.item()
                rv     = loss_r.item()
                av     = loss_a.item()
                avloss = alpha_loss.item()
                klv    = kl.item()

                # Running averages
                if running_loss is None:
                    running_loss        = lv
                    running_r_loss      = rv
                    running_a_loss      = av
                    running_alpha_loss  = avloss
                    running_kl_loss     = klv
                else:
                    running_loss       = 0.9*running_loss       + 0.1*lv
                    running_r_loss     = 0.9*running_r_loss     + 0.1*rv
                    running_a_loss     = 0.9*running_a_loss     + 0.1*av
                    running_alpha_loss = 0.9*running_alpha_loss + 0.1*avloss
                    running_kl_loss    = 0.9*running_kl_loss    + 0.1*klv

                # Append to per-epoch and global logs
                epoch_losses.append(lv)
                epoch_action_losses.append(av)
                epoch_reward_losses.append(rv)
                epoch_alpha_losses.append(avloss)
                epoch_kl_losses.append(klv)
                task_losses[tid].append(lv)
                epoch_task_losses[tid].append(lv)

                # Collect for α′ update, average acros timesteps
                # 
                task_alpha_vals[tid] = mu_alpha.mean(dim=0).detach()

                print(f"  Task {tid}: Loss={lv:.4f}, RunAvg={running_loss:.4f}, "
                      f"A-Loss={av:.4f}, R-Loss={rv:.4f}, α-Loss={avloss:.6f}")

            # After one batch_group, update α′
            all_alphas = torch.stack(list(task_alpha_vals.values()))
            new_alpha_prime     = all_alphas.mean(dim=0)
            if alpha_prime is None:
                alpha_prime = new_alpha_prime
            else:
                alpha_prime = ALPHA_P_MOMENTUM * alpha_prime + (1-ALPHA_P_MOMENTUM) * new_alpha_prime
            
            print(f"  Updated α′ norm: {alpha_prime.norm().item():.4f}")

        # Step schedulers once per epoch
        beta_scheduler.step()
        gamma_scheduler.step()
        alpha_scheduler.step()
        ll_scheduler.step()

        # Append epoch logs to global lists for plotting
        losses.extend(epoch_losses)
        action_losses.extend(epoch_action_losses)
        reward_losses.extend(epoch_reward_losses)
        alpha_losses.extend(epoch_alpha_losses)
        kl_losses.extend(epoch_kl_losses)

        print("\nTask-specific performance:")
        for tid, vals in epoch_task_losses.items():
            print(f"  Task {tid}: Avg Loss = {np.mean(vals):.4f}")
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Avg Loss:   {np.mean(epoch_losses):.4f}")
        print(f"  A-Loss:     {np.mean(epoch_action_losses):.4f}")
        print(f"  R-Loss:     {np.mean(epoch_reward_losses):.4f}")
        print(f"  α-Loss:     {np.mean(epoch_alpha_losses):.6f}")
        print(f"  Run Avg L:  {running_loss:.4f}")

    print("\nMPI-LL training complete. Generating visualizations...")
    plt.figure(figsize=(14, 10))

    # Raw losses
    plt.subplot(2, 3, 1)
    plt.plot(losses,        alpha=0.3, label="Total loss (raw)")
    plt.plot(action_losses, alpha=0.3, label="Action loss (raw)")
    plt.plot(reward_losses, alpha=0.3, label="Reward loss (raw)")
    plt.title("Raw Training Losses")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    # Moving average
    plt.subplot(2, 3, 2)
    window = min(WINDOW_SIZE, len(losses))
    if window > 0:
        ma_tot = moving_average(losses, window)
        ma_act = moving_average(action_losses, window) if action_losses else []
        ma_rwd = moving_average(reward_losses, window) if reward_losses else []
        if len(ma_tot): plt.plot(ma_tot, label="Total (MA)")
        if len(ma_act): plt.plot(ma_act, label="Action (MA)")
        if len(ma_rwd): plt.plot(ma_rwd, label="Reward (MA)")
    plt.title(f"Moving Average (w={window})")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    # Exponential moving average
    plt.subplot(2, 3, 3)
    if losses:
        ema_tot = exponential_moving_average(losses, SMOOTH_ALPHA)
        plt.plot(ema_tot, label="Total (EMA)")
    if action_losses:
        ema_act = exponential_moving_average(action_losses, SMOOTH_ALPHA)
        plt.plot(ema_act, label="Action (EMA)")
    if reward_losses:
        ema_rwd = exponential_moving_average(reward_losses, SMOOTH_ALPHA)
        plt.plot(ema_rwd, label="Reward (EMA)")
    plt.title(f"Exponential MA (α={SMOOTH_ALPHA})")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    # Per-task EMA
    plt.subplot(2, 3, 4)
    for tid, t_loss in task_losses.items():
        if t_loss:
            ema_t = exponential_moving_average(t_loss, SMOOTH_ALPHA)
            plt.plot(ema_t, label=f"Task {tid}")
    plt.title("Per-Task Loss (EMA)")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    # α-loss & KL-loss
    plt.subplot(2, 3, 5)
    if alpha_losses:
        plt.plot(exponential_moving_average(alpha_losses, SMOOTH_ALPHA), label="Alpha loss")
    if kl_losses:
        plt.plot(exponential_moving_average(kl_losses,    SMOOTH_ALPHA), label="KL loss")
    plt.title("Alpha & KL Losses (EMA)")
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)

    # Action vs Reward ratio
    plt.subplot(2, 3, 6)
    if action_losses and reward_losses:
        ratio = np.array(action_losses) / (np.array(action_losses) + np.array(reward_losses) + 1e-8)
        plt.plot(exponential_moving_average(ratio.tolist(), SMOOTH_ALPHA), label="Action/(A+R)")
        plt.axhline(0.5, color='r', linestyle='--', label="50% line")
    plt.title("Action vs Reward Contribution")
    plt.xlabel("Step"); plt.ylabel("Ratio")
    plt.legend(); plt.ylim(0,1); plt.grid(True)

    plt.tight_layout()
    plt.show()

    # only if at least one task has data
    if any(task_losses.values()):
        plt.figure(figsize=(10, 5))

        task_avg = {}
        for tid, t_loss in task_losses.items():
            if not t_loss: 
                continue
            half = len(t_loss) // 2
            task_avg[tid] = {
                'first':   np.mean(t_loss[:half]) if half>0 else 0,
                'second':  np.mean(t_loss[half:]) if half<len(t_loss) else 0,
                'overall': np.mean(t_loss),
            }

        ids       = list(task_avg.keys())
        firsts    = [task_avg[i]['first']   for i in ids]
        overalls  = [task_avg[i]['overall'] for i in ids]
        seconds   = [task_avg[i]['second']  for i in ids]

        x = np.arange(len(ids))
        w = 0.25

        plt.bar(x - w, firsts,   w, label="First half")
        plt.bar(x,     overalls, w, label="Overall")
        plt.bar(x + w, seconds,  w, label="Second half")

        plt.xticks(x, ids)
        plt.xlabel("Task ID"); plt.ylabel("Avg Loss")
        plt.title("Task Performance Comparison")
        plt.legend(); plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    else:
        print("No per-task data to visualize.")
    
    torch.save(model.state_dict(), "mpill.pt")
    print("Model saved and analysis complete.")

if __name__ == "__main__":
    main()