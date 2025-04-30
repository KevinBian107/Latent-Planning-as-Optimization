# import os
# import sys
# import torch

# os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.getcwd())

# from data import process_dataloader
# from utils.args import parse_args
# args = parse_args("configs/kitchen.yaml")
# dataloader = process_dataloader("kitchen_mixed-v2",args = args)
# args.training["num_workers"] = 0


# # for batch in dataloader:
# #     print("")



import os
import sys
import pytest
import torch
from pathlib import Path

# For direct execution (not via pytest)
def setup_path_manually():
    # Add the project root to sys.path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    os.chdir(project_root)  # Change to project root directory

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
    dataloader = process_dataloader("kitchen_mixed-v2", args=config_args)
    
    # Assertions to verify the dataloader
    assert dataloader is not None, "DataLoader should not be None"
    assert hasattr(dataloader, 'batch_size'), "DataLoader should have batch_size attribute"
    assert dataloader.batch_size == config_args.training["batch_size"], "Batch size should match configuration"
    
    print(f"✅ DataLoader created successfully with {len(dataloader)} batches")
    return dataloader

def test_dataloader_iteration(setup_path=None, config_args=None):
    """Test that we can iterate through the dataloader and access batch data."""
    if config_args is None:
        # When running directly
        from utils.args import parse_args
        config_args = parse_args("configs/kitchen.yaml")
        config_args.training["num_workers"] = 0
        
    # Get dataloader from previous test
    dataloader = test_dataloader_creation(config_args=config_args)
    
    # Test iteration (just get first batch)
    batch_count = 0
    max_batches = 2  # Only test a few batches
    
    try:
        for i, batch in enumerate(dataloader):
            batch_count += 1
            print(batch)
            
            # Check batch structure
            assert isinstance(batch, dict), "Batch should be a dictionary"
            assert len(batch) == 7, "Batch should contain 7 keys"
            assert "observations" in batch, "Batch should contain 'observations'"
            assert "actions" in batch, "Batch should contain 'actions'"
            assert "reward" in batch, "Batch should contain 'reward'"
            assert "done" in batch, "Batch should contain 'done'"
            assert "return_to_go" in batch, "Batch should contain 'return_to_go'"
            assert "prev_actions" in batch, "Batch should contain 'prev_actions'"
            assert "timesteps" in batch, "Batch should contain 'timesteps'"
            
            if i >= max_batches - 1:
                break
                
        print(f"✅ Successfully iterated through {batch_count} batches")
    except Exception as e:
        print(f"❌ Error during iteration: {e}")
        raise e

if __name__ == "__main__":
    # This allows running the test directly with python
    setup_path_manually()
    test_dataloader_creation()
    test_dataloader_iteration()
    print("All tests completed successfully!")