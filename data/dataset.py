import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader

'''---Base class for trajectory dataset---'''
class TrajectoryDataset(ABC):
    """Abstract base class for trajectory datasets."""
    
    @abstractmethod
    def get_trajectories(self):
        """
        Returns a list of trajectory dictionaries.
        Each trajectory should have at minimum:
            - observations: dict or array
            - actions: array
            - rewards: array
        """
        pass

    @property
    def is_task_organized(self):
        """
        Returns whether this dataset is already organized by task.
        
        Returns:
            bool: True if the dataset returns a task_id->trajectories dict, 
                False if it returns a list of mixed trajectories
        """
        return False


class MinariTrajectoryDataset(TrajectoryDataset):
    """Adapter for Minari datasets."""
    
    def __init__(self, dataset=None, dataset_name=None):
        """
        Initialize with either a dataset name to load or a pre-loaded dataset.
        
        Args:
            dataset_name: Name of Minari dataset to load (e.g. "D4RL/kitchen/mixed-v2")
            dataset: Pre-loaded Minari dataset
        """
        self.dataset_name = dataset_name
        self._dataset = dataset
        
        if dataset is None and dataset_name is not None:
            import minari
            self._dataset = minari.load_dataset(dataset_name, download=True)
        elif dataset is None and dataset_name is None:
            raise ValueError("Either dataset_name or dataset must be provided")

        assert hasattr(self._dataset, "__iter__") or hasattr(self._dataset, "__getitem__"), "Dataset must be iterable"
        assert hasattr(self._dataset[0], "observations"), "Dataset must have 'observations' attribute"
        assert hasattr(self._dataset[0], "actions"), "Dataset must have 'actions' attribute"
        assert hasattr(self._dataset[0], "rewards"), "Dataset must have 'rewards' attribute"
    
    def get_trajectories(self):
        """Return the list of trajectories from the Minari dataset."""
        return self._dataset
    
    @property
    def is_task_organized(self):
        return False




'''
------Below class is for future use------
'''
class CustomTrajectoryDataset(TrajectoryDataset):
    """Edit the below constructor to load a dataset other than minari."""
    
    def __init__(self, dataset):
        """
        Initialize with a list of trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
        """
        self._dataset = dataset
    
    def get_trajectories(self):
        """Return the list of trajectories."""
        return self.trajectories
    
    @property
    def is_task_organized(self):
        '''
        If task is organized, dataset should be organized such that 
        there is a key correspondign to task id and 
        there is a vlaue correspnding to the list of trajectories
        '''
        return True

