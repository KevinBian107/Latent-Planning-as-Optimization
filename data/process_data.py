from data.dataset import MinariSequenceDataset
from torch.utils.data import DataLoader

def env_mapping(env_name, env_type):
    '''
    Map the environment name to a dataset download name.
    '''
    env_name_list = env_name.split('-')

    return f"{env_type}/{env_name_list[0]}/{env_name_list[1]}-{env_name_list[2]}"


def process_dataloader(env_name,args):
    try:
        name = env_mapping(env_name, args.environment['env_type'])
        dataset = MinariSequenceDataset(name = name, context_len=args.environment['context_len'])

    except:
        raise Exception(f"Not Found environment_name {env_name}")
    
    dataloader = DataLoader(dataset, batch_size=args.training["batch_size"], shuffle=args.training["shuffle"], num_workers=args.training["num_workers"])

    return dataloader
