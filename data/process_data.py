"""
Data Processing Script for Datasets

This script processes various datasets from Gymnasium environments using Minari and saves them in a structured format.

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the CC BY-NC license found in the LICENSE.md file in the root directory of this source tree.
"""

import os
import numpy as np
import gymnasium as gym
import minari
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def split_into_trajectories(dones_float, observations, next_observations, actions, rewards):
    """
    Splits transitions into trajectories based on done signals.

    Args:
        dones_float (np.array): Array indicating episode terminations (1.0 if done, else 0.0).
        observations (np.array): Array of observations.
        next_observations (np.array): Array of next observations.
        actions (np.array): Array of actions.
        rewards (np.array): Array of rewards.

    Returns:
        List[Dict]: List of trajectories, where each trajectory is a dictionary containing observations,
                    actions, rewards, next observations, and terminals.
    """
    trajs = [defaultdict(list)]
    for i in tqdm(range(len(observations))):
        trajs[-1]['observations'].append(observations[i])
        trajs[-1]['actions'].append(actions[i])
        trajs[-1]['rewards'].append(rewards[i])
        trajs[-1]['next_observations'].append(next_observations[i])
        trajs[-1]['terminals'].append(dones_float[i])

        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append(defaultdict(list))

    for traj in trajs:
        for key in traj:
            traj[key] = np.array(traj[key])

    return trajs


def get_dataset_mean_std(dataset):
    """
    Computes the mean and standard deviation of states in the dataset.

    Args:
        dataset (Dataset): HuggingFace Dataset object containing trajectories.

    Returns:
        Tuple[np.array, np.array]: Mean and standard deviation of the states.
    """
    states = []
    for obs in dataset['observations']:
        states.extend(obs)
    states = np.vstack(states)
    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-6
    return state_mean, state_std


def extract_trajectories_from_minari(dataset):
    """
    Extracts trajectories from a Minari dataset.
    This function assumes dataset contains episodes that can be accessed by index.

    Args:
        dataset (MinariDataset): Minari dataset object.

    Returns:
        List[Dict]: List of trajectories.
    """
    trajectories = []
    
    # Get all episodes in the dataset
    total_episodes = len(dataset)
    
    for i in range(total_episodes):
        # Get episode directly by index
        episode = dataset[i]
        
        # Create trajectory data
        trajectory = {
            'observations': episode.observations,
            'actions': episode.actions,
            'rewards': episode.rewards,
            'terminals': np.zeros_like(episode.rewards)
        }
        
        # Set the last step as terminal
        if len(trajectory['terminals']) > 0:
            trajectory['terminals'][-1] = 1.0
            
        # Add next_observations field
        if hasattr(episode, 'next_observations'):
            trajectory['next_observations'] = episode.next_observations
        else:
            # If next_observations not available, create it from observations
            trajectory['next_observations'] = np.concatenate([
                episode.observations[1:],
                episode.observations[-1:] if len(episode.observations) > 0 else []
            ], axis=0)
            
        trajectories.append(trajectory)
        
    return trajectories


def process_mujoco_datasets():
    """
    Processes MuJoCo datasets and saves them to disk.
    """
    for env_name in ['halfcheetah', 'hopper', 'walker2d']:
        for dataset_type in ['medium', 'medium-replay']:
            name = f"{env_name}-{dataset_type}-v2"
            
            # Load dataset using Minari with correct naming convention
            minari_dataset_name = f"mujoco/{env_name}/{dataset_type}-v0"
            try:
                dataset = minari.load_dataset(minari_dataset_name)
                print(f"Successfully loaded: {minari_dataset_name}")
            except Exception as e:
                print(f"Error loading {minari_dataset_name}: {e}")
                print(f"Downloading dataset: {minari_dataset_name}")
                try:
                    minari.download_dataset(minari_dataset_name)
                    dataset = minari.load_dataset(minari_dataset_name)
                    print(f"Successfully downloaded and loaded: {minari_dataset_name}")
                except Exception as download_error:
                    print(f"Failed to download {minari_dataset_name}: {download_error}")
                    print(f"Skipping {name}")
                    continue
            
            # Extract trajectories
            try:
                paths = extract_trajectories_from_minari(dataset)
                
                returns = np.array([np.sum(p['rewards']) for p in paths])
                num_samples = np.sum([len(p['rewards']) for p in paths])
                print(f"Processing {name}:")
                print(f"Number of samples collected: {num_samples}")
                print(f"Number of trajectories: {len(paths)}")
                print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
                    f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")

                # Prepare dataset for saving
                dataset_list = [{
                    'observations': traj['observations'],
                    'actions': traj['actions'],
                    'rewards': traj['rewards'],
                    'dones': traj['terminals']
                } for traj in paths]

                dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
                dataset_hf = Dataset.from_dict(dataset_dict)
                dataset_hf = DatasetDict({'train': dataset_hf})
                print(dataset_hf)

                # Save dataset to disk
                directory_path = f'{name}'
                os.makedirs(directory_path, exist_ok=True)
                dataset_hf.save_to_disk(directory_path)
            except Exception as process_error:
                print(f"Error processing {name}: {process_error}")
                continue


def process_antmaze_datasets():
    """
    Processes AntMaze datasets and saves them to disk.
    """
    env_mapping = {
        'antmaze-umaze-v2': 'D4RL/antmaze/umaze-v1',
        'antmaze-medium-diverse-v2': 'D4RL/antmaze/medium-diverse-v1',
        'antmaze-large-diverse-v2': 'D4RL/antmaze/large-diverse-v1',
    }
    
    for env_name, minari_name in env_mapping.items():
        # Load dataset using Minari
        try:
            dataset = minari.load_dataset(minari_name)
            print(f"Successfully loaded: {minari_name}")
        except Exception as e:
            print(f"Error loading {minari_name}: {e}")
            print(f"Downloading dataset: {minari_name}")
            try:
                minari.download_dataset(minari_name)
                dataset = minari.load_dataset(minari_name)
                print(f"Successfully downloaded and loaded: {minari_name}")
            except Exception as download_error:
                print(f"Failed to download {minari_name}: {download_error}")
                print(f"Skipping {env_name}")
                continue
        
        # Extract trajectories
        try:
            trajectories = extract_trajectories_from_minari(dataset)

            returns = np.array([np.sum(traj['rewards']) for traj in trajectories])
            lengths = np.array([len(traj['rewards']) for traj in trajectories])
            num_samples = np.sum(lengths)
            print(f"Processing {env_name}:")
            print(f"Number of samples collected: {num_samples}")
            print(f"Number of trajectories: {len(trajectories)}")
            print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
                f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")
            print(f"Trajectory lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
                f"max={np.max(lengths)}, min={np.min(lengths)}")

            # Prepare dataset for saving
            dataset_list = [{
                'observations': traj['observations'],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'dones': traj['terminals']
            } for traj in trajectories]

            dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
            dataset_hf = Dataset.from_dict(dataset_dict)
            dataset_hf = DatasetDict({'train': dataset_hf})
            print(dataset_hf)

            # Save dataset to disk
            directory_path = f'dataset/{env_name}'
            os.makedirs(directory_path, exist_ok=True)
            dataset_hf.save_to_disk(directory_path)
        except Exception as process_error:
            print(f"Error processing {env_name}: {process_error}")
            continue


def process_maze2d_datasets():
    """
    Processes Maze2D datasets and saves them to disk.
    """
    env_mapping = {
        'maze2d-umaze-v1': 'D4RL/pointmaze/umaze-v2',
        'maze2d-medium-v1': 'D4RL/pointmaze/medium-v2',
        'maze2d-large-v1': 'D4RL/pointmaze/large-v2',
    }
    
    info = {}
    for env_name, minari_name in env_mapping.items():
        print(f"Processing {env_name}:")
        
        # Load dataset using Minari
        try:
            dataset = minari.load_dataset(minari_name)
            print(f"Successfully loaded: {minari_name}")
        except Exception as e:
            print(f"Error loading {minari_name}: {e}")
            print(f"Downloading dataset: {minari_name}")
            try:
                minari.download_dataset(minari_name)
                dataset = minari.load_dataset(minari_name)
                print(f"Successfully downloaded and loaded: {minari_name}")
            except Exception as download_error:
                print(f"Failed to download {minari_name}: {download_error}")
                print(f"Skipping {env_name}")
                continue
        
        # Extract trajectories
        try:
            trajectories = extract_trajectories_from_minari(dataset)

            returns = np.array([np.sum(traj['rewards']) for traj in trajectories])
            lengths = np.array([len(traj['rewards']) for traj in trajectories])
            num_samples = np.sum(lengths)
            num_nonzero_tot_rew = np.sum(returns != 0)
            print(f"Number of samples collected: {num_samples}")
            print(f"Number of trajectories: {len(trajectories)}")
            print(f"Number of non-zero return trajectories: {num_nonzero_tot_rew}")
            print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
                f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")
            print(f"Trajectory lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
                f"max={np.max(lengths)}, min={np.min(lengths)}")
            print("-" * 30)

            # Prepare dataset for saving
            dataset_list = [{
                'observations': traj['observations'],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'dones': traj['terminals']
            } for traj in trajectories]

            dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
            dataset_hf = Dataset.from_dict(dataset_dict)
            dataset_hf = DatasetDict({'train': dataset_hf})
            print(dataset_hf)

            # Save dataset to disk
            directory_path = f'{env_name}'
            os.makedirs(directory_path, exist_ok=True)
            dataset_hf.save_to_disk(directory_path)

            # Compute mean and std of states
            state_mean, state_std = get_dataset_mean_std(dataset_hf['train'])
            info[env_name] = {'state_mean': state_mean, 'state_std': state_std}
        except Exception as process_error:
            print(f"Error processing {env_name}: {process_error}")
            continue

    print(info)


def process_kitchen_datasets():
    """
    Processes Kitchen datasets and saves them to disk.
    """
    env_mapping = {
        'kitchen-complete-v0': 'D4RL/kitchen/complete-v2',
        'kitchen-partial-v0': 'D4RL/kitchen/partial-v2',
        'kitchen-mixed-v0': 'D4RL/kitchen/mixed-v2'
    }
    
    info = {}
    for env_name, minari_name in env_mapping.items():
        # Load dataset using Minari
        try:
            dataset = minari.load_dataset(minari_name)
            print(f"Successfully loaded: {minari_name}")
        except Exception as e:
            print(f"Error loading {minari_name}: {e}")
            print(f"Downloading dataset: {minari_name}")
            try:
                minari.download_dataset(minari_name)
                dataset = minari.load_dataset(minari_name)
                print(f"Successfully downloaded and loaded: {minari_name}")
            except Exception as download_error:
                print(f"Failed to download {minari_name}: {download_error}")
                print(f"Skipping {env_name}")
                continue
        
        # Extract trajectories
        try:
            trajectories = extract_trajectories_from_minari(dataset)

            print(f"Processing {env_name}:")
            returns = np.array([np.sum(traj['rewards']) for traj in trajectories])
            lengths = np.array([len(traj['rewards']) for traj in trajectories])
            num_samples = np.sum(lengths)
            print(f"Number of samples collected: {num_samples}")
            print(f"Number of trajectories: {len(trajectories)}")
            print(f"Trajectory returns: mean={np.mean(returns):.2f}, std={np.std(returns):.2f}, "
                f"max={np.max(returns):.2f}, min={np.min(returns):.2f}")
            print(f"Trajectory lengths: mean={np.mean(lengths):.2f}, std={np.std(lengths):.2f}, "
                f"max={np.max(lengths)}, min={np.min(lengths)}")

            # Prepare dataset for saving
            dataset_list = [{
                'observations': traj['observations'],
                'actions': traj['actions'],
                'rewards': traj['rewards'],
                'dones': traj['terminals']
            } for traj in trajectories]

            dataset_dict = {key: [d[key] for d in dataset_list] for key in dataset_list[0]}
            dataset_hf = Dataset.from_dict(dataset_dict)
            dataset_hf = DatasetDict({'train': dataset_hf})
            print(dataset_hf)

            # Save dataset to disk
            directory_path = f'{env_name}'
            os.makedirs(directory_path, exist_ok=True)
            dataset_hf.save_to_disk(directory_path)

            # Compute mean and std of states
            state_mean, state_std = get_dataset_mean_std(dataset_hf['train'])
            info[env_name] = {'state_mean': state_mean, 'state_std': state_std}
        except Exception as process_error:
            print(f"Error processing {env_name}: {process_error}")
            continue

    print(info)


if __name__ == '__main__':
    process_mujoco_datasets()
    process_antmaze_datasets()
    process_maze2d_datasets()
    process_kitchen_datasets()
