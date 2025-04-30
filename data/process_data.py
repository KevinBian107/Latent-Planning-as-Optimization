from data.dataset import MinariSequenceDataset
from torch.utils.data import DataLoader


def process_dataloader(env_name,args):
    if env_name == 'kitchen_mixed-v2':
        dataset = MinariSequenceDataset(name = 'D4RL/kitchen/mixed-v2', context_len=args.environment['context_len'])
        dataloader = DataLoader(dataset, batch_size=args.training["batch_size"], shuffle=args.training["shuffle"], num_workers=args.training["num_workers"])
    elif env_name == 'kitchen_complete-v2':
        dataset = MinariSequenceDataset(name = 'D4RL/kitchen/complete-v2', context_len=args.environment['context_len'])
        dataloader = DataLoader(dataset, batch_size=args.training["batch_size"], shuffle=args.training["shuffle"], num_workers=args.training["num_workers"])
    else:
        raise Exception(f"Not Found environment_name {env_name}")
    return dataloader
        