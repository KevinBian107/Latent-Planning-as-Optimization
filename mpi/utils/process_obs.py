## This is used for process the observation dict in Franka kitchen / Maze2d and so on.
import torch
import numpy as np


kitchen_goal_obs_dict = {
    "bottom burner": {"type": "slide", "dim": 2},
    "top burner": {"type": "slide", "dim": 2},
    "light switch": {"type": "slide", "dim": 2},
    "slide cabinet": {"type": "slide", "dim": 1},
    "hinge cabinet": {"type": "hinge", "dim": 2},
    "microwave": {"type": "hinge", "dim": 1},
    "kettle": {"type": "free", "dim": 7},
}


def pad_goal_dict(goal_obs_dict, goal_dict_seq):
    """
    兼容 单步 (dict[str -> np.array(dim,)])
         批量 (dict[str -> np.array(T, dim)])
    """
    # 先找第一个 key
    any_key = next(iter(goal_obs_dict.keys()))
    arr = goal_dict_seq.get(any_key, None)
    # 判断是否为 None、以及是否为一维
    if arr is not None and len(np.shape(arr)) == 2:
        # 多步
        T = arr.shape[0]
        padded_chunks = []
        for name, cfg in goal_obs_dict.items():
            dim = cfg['dim']
            if name in goal_dict_seq:
                data = np.asarray(goal_dict_seq[name], dtype=np.float32)
                D_actual = data.shape[1]
                pad = np.zeros((T, dim), dtype=np.float32)
                pad[:, :min(dim, D_actual)] = data[:, :min(dim, D_actual)]
            else:
                pad = np.zeros((T, dim), dtype=np.float32)
            padded_chunks.append(pad)
        return torch.tensor(np.concatenate(padded_chunks, axis=1), dtype=torch.float32)
    else:
        # 单步
        padded_chunks = []
        for name, cfg in goal_obs_dict.items():
            dim = cfg['dim']
            if name in goal_dict_seq:
                data = np.asarray(goal_dict_seq[name], dtype=np.float32)
                D_actual = data.shape[0]
                pad = np.zeros(dim, dtype=np.float32)
                pad[:min(dim, D_actual)] = data[:min(dim, D_actual)]
            else:
                pad = np.zeros(dim, dtype=np.float32)
            padded_chunks.append(pad)
        return torch.tensor(np.concatenate(padded_chunks), dtype=torch.float32)


def process_observation(goal_obs_dict, episode):
    """
    Inputs:
        goal_obs_dict: describes expected keys and dims
        episode: dict with keys ['observation', 'desired_goal', 'achieved_goal']
    Returns:
        dict with same keys but all values are torch.Tensor with shape (T, *)
    """
    return {
        'observation': torch.tensor(episode['observation'], dtype=torch.float32),
        'desired_goal': pad_goal_dict(goal_obs_dict, episode['desired_goal']),
        'achieved_goal': pad_goal_dict(goal_obs_dict, episode['achieved_goal']),
    }

if __name__ == "__main__":
    obs_dict = {
    'observation': np.random.rand(30),  # already flat np.array
    'desired_goal': {
        'bottom burner': [-0.8, 0.0],
        'kettle': [-0.2, 0.8, 1.6, 1.0, 0., 0., -0.05]
    },
    'achieved_goal': {
        'bottom burner': [-0.78, 0.01],
        'kettle': [-0.21, 0.76, 1.63, 1.01, 0., 0., -0.04]
    }
    }

    result = process_observation(kitchen_goal_obs_dict, obs_dict)

    print(result)
