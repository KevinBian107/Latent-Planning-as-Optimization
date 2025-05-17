import minari
from tqdm import tqdm

def download_data(minari_names):
    """
    Download datasets from Minari.
    
    Args:
        minari_names (list): List of dataset names to download.
    """
    for minari_name in tqdm(minari_names, desc="Downloading datasets"):
        # Download the dataset
        print(f"Downloading dataset {minari_name}...")
        minari.download_dataset(minari_name)

if __name__ == "__main__":
    minari_names = [
        "mujoco/halfcheetah/medium-v0", 
        "mujoco/hopper/medium-v0", 
        "mujoco/walker2d/medium-v0", 
        "D4RL/antmaze/umaze-v1",
        "D4RL/antmaze/medium-diverse-v1",
        "D4RL/antmaze/large-diverse-v1",
        "D4RL/pointmaze/umaze-v2",
        "D4RL/pointmaze/medium-v2",
        "D4RL/pointmaze/large-v2",
        "D4RL/kitchen/complete-v2",
        "D4RL/kitchen/partial-v2",
        "D4RL/kitchen/mixed-v2"
    ]

    download_data(minari_names)
    print("✅ All datasets downloaded successfully.")
