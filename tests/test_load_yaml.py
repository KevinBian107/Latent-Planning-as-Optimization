import os
import sys
import pytest
from pathlib import Path

# For python execution
def setup_path_manually():
    # Add the project root to sys.path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    # Change current working directory to project root for consistent path resolution
    os.chdir(project_root)

# For pytest execution
@pytest.fixture(scope="session", autouse=True) # autouse=True ensures it runs for all tests in the session
def setup_path():
    setup_path_manually()

# Find all YAML files in the configs directory
CONFIG_DIR = Path("configs")
YAML_FILES = list(CONFIG_DIR.glob("*.yaml"))
YAML_FILE_PATHS = [str(f) for f in YAML_FILES if f.is_file()]

@pytest.mark.parametrize("config_path", YAML_FILE_PATHS)
def test_load_yaml_config(config_path):
    """Test that the YAML config file loads correctly."""
    from utils.args import load_yaml_config

    print(f"\nTesting config file: {config_path}")
    # Load the configuration
    try:
        config = load_yaml_config(config_path)
    except Exception as e:
        pytest.fail(f"Failed to load YAML file {config_path}: {e}")

    # Verify the configuration has expected top-level keys
    assert isinstance(config, dict), f"Config from {config_path} should be a dictionary"
    assert "path" in config, f"Config {config_path} should have a 'path' key"
    assert "training" in config, f"Config {config_path} should have a 'training' key"
    assert "environment" in config, f"Config {config_path} should have an 'environment' key"

    # Verify nested keys (add more specific checks as needed)
    assert "checkpoint_path" in config['path'], f"'path' in {config_path} missing 'checkpoint_path'"
    assert "weights_path" in config['path'], f"'path' in {config_path} missing 'weights_path'"
    assert "logs_path" in config['path'], f"'path' in {config_path} missing 'logs_path'"
    assert "videos_path" in config['path'], f"'path' in {config_path} missing 'videos_path'"

    assert "batch_size" in config['training'], f"'training' in {config_path} missing 'batch_size'"
    assert "learning_rate" in config['training'], f"'training' in {config_path} missing 'learning_rate'"
    assert "epochs" in config['training'], f"'training' in {config_path} missing 'epoch'"

    assert "name" in config['environment'], f"'environment' in {config_path} missing 'name'"
    assert "env_type" in config['environment'], f"'environment' in {config_path} missing 'env_type'"
    assert "context_len" in config['environment'], f"'environment' in {config_path} missing 'context_len'"

    print(f"✅ Configuration loaded successfully from {config_path}")

if __name__ == "__main__":
    # This allows running the test directly with python
    setup_path_manually()
    print(f"Found {len(YAML_FILE_PATHS)} YAML files to test in {CONFIG_DIR}:")
    all_passed = True
    for yaml_file in YAML_FILE_PATHS:
        try:
            test_load_yaml_config(yaml_file)
        except Exception as e:
            print(f"❌ Test failed for {yaml_file}: {e}")
            all_passed = False
            # Optionally re-raise or break if you want to stop on first failure
            # raise e
            # break

    if all_passed:
        print("\n✅ All YAML configuration tests passed!")
    else:
        print("\n❌ Some YAML configuration tests failed.")
        sys.exit(1) # Exit with a non-zero code to indicate failure

