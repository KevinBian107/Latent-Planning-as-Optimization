# import os
# import sys
# os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(os.getcwd())
# from utils.args import load_yaml_config

# print(load_yaml_config("configs/basic.yaml"))


import os
import sys
import pytest
from pathlib import Path

# For python execution
def setup_path_manually():
    # Add the project root to sys.path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
# For pytest execution
@pytest.fixture(scope="session")
def setup_path():
    setup_path_manually()

def test_load_yaml_config():
    """Test that the YAML config file loads correctly."""
    from utils.args import load_yaml_config
    
    # Load the configuration
    config = load_yaml_config("configs/basic.yaml")
    
    # Verify the configuration has expected fields
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "path" in config, "Config should have a 'path' key"
    assert "training" in config, "Config should have a 'training' key"
    assert "environment" in config, "Config should have a 'environment' key"

    print("✅ Configuration loaded successfully")
    return None


if __name__ == "__main__":
    # This allows running the test directly with python
    setup_path_manually()
    test_load_yaml_config()