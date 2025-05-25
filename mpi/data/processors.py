"""
Available processors in this module:
- SequenceProcessor: Processes trajectory segments into fixed-length sequences.
- KitchenSegmenter: Segments kitchen trajectories into task-specific segments.
- ... (add more processors as needed)
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
import pdb


class BaseProcessor(ABC):
    """Abstract base class for all processors in the pipeline."""

    @abstractmethod
    def process(self, data):
        """
        Process the input data and return processed data.

        Args:
            data: Input data of appropriate type for this processor
            
        Returns:
            Processed data
        """
        pass


class SequenceProcessor(BaseProcessor):
    """
    Processes trajectory segments into fixed-length sequences for training.
    """
    def __init__(self, context_len, device="cpu"):
        """
        Initialize the processor.
        
        Args:
            max_len (int): Horizon length in LPT setup
        """
        self.max_len = context_len
        self.device = device
        self.sequence_data = []
    
    def process(self, data):
        return self.process_episode(data)

    def process_episode(self, episode):
        """
        Process a single episode into training sequences.
        
        Args:
            episode (dict): Episode data containing observations, actions, rewards
            
        Returns:
            list: List of processed sequence dictionaries
        
        Information input in a sequence:
        - observations: (T, obs_dim)
        - actions: (T, act_dim)
        - rewards: (T, 1)
        
        Information output in a sequence:
        - observations: (context_len, obs_dim)
        - actions: (context_len, act_dim)
        - reward: (context_len, 1)
        - return_to_go: (context_len, 1)
        - prev_actions: (context_len, act_dim)
        - timesteps: (context_len, 1)
        """
        # Handle both array and dictionary observations
        # pdb.set_trace()
        # if isinstance(episode['observations'], dict) and 'observation' in episode['observations']:
        #     obs = torch.tensor(episode['observations']["observation"][:-1], dtype=torch.float32)
        # else:
        #     # Fallback to direct observations if observation key doesn't exist
        #     obs = torch.tensor(episode['observations'][:-1], dtype=torch.float32)
        # acts = torch.tensor(episode['actions'], dtype=torch.float32)
        # rews = torch.tensor(episode['rewards'], dtype=torch.float32)
        
        # Check if episode is an object or a dictionary
        if hasattr(episode, 'observations'):
            # Episode is an object with attributes
            if isinstance(episode.observations, dict) and 'observation' in episode.observations:
                obs = torch.tensor(episode.observations["observation"][:-1], dtype=torch.float32)
            else:
                # Fallback to direct observations if observation key doesn't exist
                obs = torch.tensor(episode.observations[:-1], dtype=torch.float32)
                
            acts = torch.tensor(episode.actions, dtype=torch.float32)
            rews = torch.tensor(episode.rewards, dtype=torch.float32)
        else:
            # Episode is a dictionary (which is the case for "segmeneter"-based dataset)
            if isinstance(episode['observations'], dict) and 'observation' in episode['observations']:
                obs = torch.tensor(episode['observations']["observation"][:-1], dtype=torch.float32)
            else:
                # Fallback to direct observations if observation key doesn't exist
                obs = torch.tensor(episode['observations'][:-1], dtype=torch.float32)
                
            acts = torch.tensor(episode['actions'], dtype=torch.float32)
            rews = torch.tensor(episode['rewards'], dtype=torch.float32)

        # if isinstance(episode.observations, dict) and 'observation' in episode.observations:
        #     obs = torch.tensor(episode.observations.observation[:-1], dtype=torch.float32)
        # else:
        #     # Fallback to direct observations if observation key doesn't exist
        #     obs = torch.tensor(episode.observations[:-1], dtype=torch.float32)
        
        # acts = torch.tensor(episode.actions, dtype=torch.float32)
        # rews = torch.tensor(episode.rewards, dtype=torch.float32)
        
        # compute more information:
        rtg = rews.flip([0]).cumsum(0).flip([0]).unsqueeze(-1)
        prev_acts = torch.cat([torch.zeros_like(acts[:1]), acts[:-1]], dim=0)
        timesteps = torch.arange(len(obs)).unsqueeze(-1)
        
        sequences = []
        
        # If the episode is shorter than max_len, pad it with 0
        if obs.shape[0] < self.max_len:
            # pad the sequence to max_len
            pad_len = self.max_len - obs.shape[0]
            obs = torch.cat([obs, torch.zeros(pad_len, obs.shape[1], dtype=obs.dtype)], dim=0)
            acts = torch.cat([acts, torch.zeros(pad_len, acts.shape[1], dtype=acts.dtype)], dim=0)
            rews = torch.cat([rews, torch.zeros(pad_len, dtype=rews.dtype)], dim=0)
            rtg = torch.cat([rtg, torch.zeros(pad_len, 1, dtype=rtg.dtype)], dim=0)
            prev_acts = torch.cat([prev_acts, torch.zeros(pad_len, prev_acts.shape[1], dtype=prev_acts.dtype)], dim=0)
            timesteps = torch.cat([timesteps, torch.zeros(pad_len, 1, dtype=timesteps.dtype)], dim=0)

            # add the padded sequence
            sequences.append({
                "observations": obs, 
                "actions": acts,
                "reward": rews.unsqueeze(-1), 
                "return_to_go": rtg, 
                "prev_actions": prev_acts,
                "timesteps": timesteps,
            })
            return sequences
        
        # If longer than max_len, create sliding windows. 
        # there will be (num_time_step - max_len + 1) number of sequences
        for i in range(obs.shape[0] - self.max_len + 1):
            sequences.append({
                "observations": obs[i:i+self.max_len],
                "actions": acts[i:i+self.max_len],
                "reward": rews[i:i+self.max_len].unsqueeze(-1),
                "return_to_go": rtg[i:i+self.max_len],
                "prev_actions": prev_acts[i:i+self.max_len],
                "timesteps": timesteps[i:i+self.max_len],
            })
        
        return sequences


class KitchenSegmenter(BaseProcessor):
    """
    Segments kitchen trajectories into task-specific segments.
    """
    def __init__(
            self, task_goal_keys, proximity_thresholds, stability_duration
        ):
        """
        Args:
            task_goal_keys: List of keys for tasks to detect
            proximity_thresholds: Dict mapping task_key to proximity threshold
            stability_duration: Number of timesteps a goal must remain close
        """
        self.task_goal_keys = task_goal_keys
        self.proximity_thresholds = proximity_thresholds
        self.stability_duration = stability_duration
    
    def process(self, data):
        return self.segment_trajectory(data)
    
    def segment_trajectory(self, trajectory):
        """
        Segments a single long trajectory into multiple sub-trajectories based on
        stable completion of sub-tasks. A sub-task is complete when its achieved_goal 
        stays close to its desired_goal for a specified stability_duration.

        Args:
            full_episode: A dictionary-like object representing one full episode.
                        - full_episode.observations['achieved_goal'][task_key] is a sequence (step, D_task_goal_dim)
                        - full_episode.observations['desired_goal'][task_key] is a static target (step, D_task_goal_dim)
        Returns:
            A list of dictionaries, where each dictionary is a segmented sub-trajectory.
            Each sub-trajectory will have an 'task_id' field.
        """
        segmented_trajectories = []

        all_obs_data = trajectory.observations
        all_actions = trajectory.actions
        all_rewards = trajectory.rewards
        all_terminations = trajectory.terminations
        all_truncations = trajectory.truncations

        num_total_steps = len(all_actions)
        if num_total_steps == 0 or self.stability_duration <= 0:
            return []

        current_segment_start_idx = 0
        
        # Tracks consecutive timesteps each task has been "close"
        task_close_streaks = {key: 0 for key in self.task_goal_keys}
        
        # Tracks which of the main tasks have been segmented
        tasks_segmented_this_episode = set()

        for t in range(num_total_steps):
            task_to_segment = None

            for task_key in self.task_goal_keys:
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

                prox_threshold = self.proximity_thresholds.get(task_key)
                if prox_threshold is None:
                    raise ValueError(f"Proximity threshold for task '{task_key}' not provided.")
                    # if task_key in task_close_streaks: del task_close_streaks[task_key]
                    # continue

                if diff < prox_threshold:
                    task_close_streaks[task_key] += 1
                else:
                    task_close_streaks[task_key] = 0

                # Check if this task met the stability duration
                if task_close_streaks[task_key] >= self.stability_duration:
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
                task_close_streaks = {key: 0 for key in self.task_goal_keys if key not in tasks_segmented_this_episode}

                if not task_close_streaks: 
                    break
        
        remaining_task = list(set(self.task_goal_keys) - tasks_segmented_this_episode)
        if len(remaining_task) > 0:
            remaining_task = remaining_task[0]
        else:
            remaining_task = None

        # Handle any remaining part of the trajectory
        if current_segment_start_idx < num_total_steps and len(tasks_segmented_this_episode) < len(self.task_goal_keys):
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


