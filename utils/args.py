import yaml
import argparse

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)



def parse_args(args: argparse.Namespace) -> argparse.Namespace:
    # 1. Load config from YAML
    yaml_config = load_yaml_config(args.config)

    # 2. Convert args to a mutable dict
    args_dict = vars(args).copy()

    # 3. Merge YAML fields (preserve args values if not None)
    for key, value in yaml_config.items():
        if key not in args_dict or args_dict[key] is None:
            args_dict[key] = value

    # 4. Return new Namespace with all combined fields
    return argparse.Namespace(**args_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kitchen.yaml")
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = parse_args(args)

    print(config["device"])
    print(config["training"]["batch_size"])