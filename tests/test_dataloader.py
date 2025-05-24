import os
import sys
import pytest
import torch
from pathlib import Path

# For direct execution
def setup_path_manually():
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    os.chdir(project_root)

# For pytest execution
@pytest.fixture(scope="session")
def setup_path():
    setup_path_manually()
    
@pytest.fixture
def config_args():
    from utils.args import parse_args
    args = parse_args("configs/kitchen.yaml")
    # Set num_workers to 0 to avoid multiprocessing issues in tests
    args.training["num_workers"] = 0
    return args

def test_dataloader_creation(setup_path=None, config_args=None):
    """Test that the dataloader can be created successfully."""
    if config_args is None:
        # When running directly
        from utils.args import parse_args
        config_args = parse_args("configs/kitchen.yaml")
        config_args.training["num_workers"] = 0
        
    from data import process_dataloader
    
    # Create the dataloader
    dataloader = process_dataloader("kitchen-mixed-v2", args=config_args)
    
    # Assertions to verify the dataloader
    assert dataloader is not None, "DataLoader should not be None"
    assert hasattr(dataloader, 'batch_size'), "DataLoader should have batch_size attribute"
    assert dataloader.batch_size == config_args.training["batch_size"], "Batch size should match configuration"
    
    print(f"✅ DataLoader created successfully with {len(dataloader)} batches")

def test_dataloader_iteration(setup_path=None, config_args=None):
    """Test that we can iterate through the dataloader and access batch data."""
    if config_args is None:
        # When running directly
        from utils.args import parse_args
        config_args = parse_args("configs/kitchen.yaml")
        config_args.training["num_workers"] = 0
    
    from data import process_dataloader

    # Get dataloader from previous test
    dataloader = process_dataloader("kitchen-mixed-v2", args=config_args)
    
    # Test iteration
    batch_count = 0
    max_batches = 2 
    
    try:
        for i, batch in enumerate(dataloader):
            batch_count += 1
            
            # Check batch structure
            assert isinstance(batch, dict), "Batch should be a dictionary"
            assert len(batch) == 7, "Batch should contain 7 keys"
            
            # check shape
            assert "observations" in batch, "Batch should contain 'observations'"
            assert batch['observations'].shape[:2] == (
                config_args.training['batch_size'], config_args.environment['context_len']
            ), "Observations shape mismatch"

            assert "actions" in batch, "Batch should contain 'actions'"
            assert batch['actions'].shape[:2] == (
                config_args.training['batch_size'], config_args.environment['context_len']
            ), "Actions shape mismatch"

            assert "reward" in batch, "Batch should contain 'reward'"
            assert batch['reward'].shape == (
                config_args.training['batch_size'], config_args.environment['context_len'], 1
            ), "Reward shape mismatch"

            assert "done" in batch, "Batch should contain 'done'"
            assert batch['done'].shape == (
                config_args.training['batch_size'], config_args.environment['context_len'], 1
            ), "Done shape mismatch"
            
            assert "return_to_go" in batch, "Batch should contain 'return_to_go'"
            assert batch['return_to_go'].shape == (
                config_args.training['batch_size'], config_args.environment['context_len'], 1
            ), "Return to go shape mismatch"

            assert "prev_actions" in batch, "Batch should contain 'prev_actions'"
            assert batch['prev_actions'].shape[:2] == (
                config_args.training['batch_size'], config_args.environment['context_len']
            ), "Previous actions shape mismatch"

            assert "timesteps" in batch, "Batch should contain 'timesteps'"
            assert batch['timesteps'].shape == (
                config_args.training['batch_size'], config_args.environment['context_len'], 1
            ), "Timesteps shape mismatch"

            if i >= max_batches - 1:
                break
                
        print(f"✅ Successfully iterated through {batch_count} batches")
    except Exception as e:
        print(f"❌ Error during iteration: {e}")
        raise e

if __name__ == "__main__":
    setup_path_manually()
    test_dataloader_creation()
    test_dataloader_iteration()
    print("All tests completed successfully!")