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
    # dataset = minari.load_dataset("D4RL/kitchen/mixed-v2", download=True)
    dataset = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
    
    # task_datasets = organize_data_by_task(dataset)
    task_datasets = split_task(dataset)

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

    torch.save(model.state_dict(), "mpill.pt")
    print("Model saved and analysis complete.")

if __name__ == "__main__":
    main()