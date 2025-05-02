import torch
from torch.utils.data import DataLoader, Dataset
import minari
from tqdm import tqdm

class MinariSequenceDataset(Dataset):
    def __init__(self, context_len, name, device="cpu"):
        self.context_len = context_len
        self.device = device
        self.sequence_data = []
        self.name = name
        self._load()

    def _load(self):
        dataset = minari.load_dataset(self.name, download=True)
        for episode in tqdm(dataset, desc="Processing episodes"):
            obs = torch.tensor(episode.observations["observation"][:-1], dtype=torch.float32).to(self.device)
            actions = torch.tensor(episode.actions, dtype=torch.float32).to(self.device)
            rewards = torch.tensor(episode.rewards, dtype=torch.float32).to(self.device)
            dones = torch.tensor(episode.terminations, dtype=torch.bool).to(self.device)

            rtg = rewards.flip(dims=[0]).cumsum(dim=0).flip(dims=[0]).unsqueeze(-1)
            prev_act = torch.cat([torch.zeros_like(actions[:1]), actions[:-1]], dim=0)
            timesteps = torch.arange(len(obs), dtype=torch.long, device=self.device).unsqueeze(-1)

            if obs.shape[0] < self.context_len:
                continue

            for i in range(obs.shape[0] - self.context_len + 1):
                self.sequence_data.append({
                    "observations": obs[i:i+self.context_len],
                    "actions": actions[i:i+self.context_len],
                    "reward": rewards[i:i+self.context_len].unsqueeze(-1),
                    "done": dones[i:i+self.context_len].unsqueeze(-1),
                    "return_to_go": rtg[i:i+self.context_len],
                    "prev_actions": prev_act[i:i+self.context_len],
                    "timesteps": timesteps[i:i+self.context_len],
                })

        print(f"✅ Loaded {len(self.sequence_data)} sequences.")

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        return self.sequence_data[idx]
