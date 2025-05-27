'''
Available batch generators in this module:
- TaskBatchGenerator: Generates batches for task-specific data.
- SingleTaskBatchGenerator: Generates batches for a single task.
...

'''


import numpy as np
import torch
from abc import ABC, abstractmethod

class BatchGenerator(ABC):
    @abstractmethod
    def get_batch(self, data):
        """
        Generate batches from the data.
        
        Args:
            data: Data to be batched
            
        Yields:
            Dict of batched tensors, each with shape [batch_size, ...]
        """
        pass


class TaskBatchGenerator(BatchGenerator):
    """Generates batches for task-specific data."""
    def __init__(self, processed_data, batch_size, device='cpu', shuffle=True):
        """
        Args:
            batch_size: Number of sequences per batch
            device: Device to place tensors on
            shuffle: Whether to shuffle data
        """
        self.processed_data = processed_data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle

    def get_batch(self, task_id):
        """
        Generate batches for task-specific sequence data.
        
        Args:
            task_data: List of processed sequence dictionaries
            
        Yields:
            Dict of batched tensors, each with shape [batch_size, ...]
        """
        assert isinstance(self.processed_data, dict), "Processed data should be a dictionary with task IDs as keys."
        assert task_id in self.processed_data, f"Task ID {task_id} not found in processed data."
        
        task_data = self.processed_data[task_id]

        if len(task_data) < self.batch_size:
            # Single batch with repeat sampling
            indices = np.random.choice(
                len(task_data), 
                size=self.batch_size, 
                replace=True
            )
            batch = [task_data[idx] for idx in indices]
            yield {k: torch.stack([d[k] for d in batch]).to(self.device) 
                  for k in batch[0]}
        else:
            # Multiple batches
            indices = np.arange(len(task_data))
            if self.shuffle:
                np.random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                
                # Fill remainder of last batch by sampling with replacement
                if len(batch_indices) < self.batch_size:
                    extra = np.random.choice(
                        len(task_data), 
                        size=self.batch_size-len(batch_indices), 
                        replace=True
                    )
                    batch_indices = np.concatenate([batch_indices, extra])
                
                # Create batch
                batch = [task_data[idx % len(task_data)] for idx in batch_indices]
                yield {k: torch.stack([d[k] for d in batch]).to(self.device) 
                      for k in batch[0]}




class SingleTaskBatchGenerator(BatchGenerator):
    """Generates batches for task-specific data."""
    def __init__(self, processed_data, batch_size, device='cpu', shuffle=True):
        """
        Args:
            batch_size: Number of sequences per batch
            device: Device to place tensors on
            shuffle: Whether to shuffle data
        """
        self.processed_data = processed_data
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
    
    def get_batch(self):
        """
        Generate batches for single task sequence data.
        
        Yields:
            Dict of batched tensors, each with shape [batch_size, ...]
        """
        assert isinstance(self.processed_data, list), "Processed data should be a list of sequence dictionaries."
        
        indices = np.arange(len(self.processed_data))

        if self.shuffle:
            np.random.shuffle(self.processed_data)
        
        for i in range(0, len(self.processed_data), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            
            # Fill remainder of last batch if needed
            if len(batch_indices) < self.batch_size:
                extra = np.random.choice(
                    len(self.processed_data), 
                    self.batch_size-len(batch_indices), 
                    replace=True
                )
                batch_indices = np.concatenate([batch_indices, extra])
            
            # Create batch
            batch = [self.processed_data[idx % len(self.processed_data)] for idx in batch_indices]
            
            yield {k: torch.stack([d[k] for d in batch]).to(self.device) 
                   for k in batch[0]}


    def get_random_batch(self):
        '''
        Get one random batch of data from the processed data.
        
        CAUTION: No all data will be used 
        '''
        assert isinstance(self.processed_data, list), "Processed data should be a list of sequence dictionaries."
        
        # Handle case where data is smaller than batch size
        if len(self.processed_data) < self.batch_size:
            idxs = np.random.choice(
                len(self.processed_data), 
                self.batch_size, 
                replace=True
            )
        else:
            # Random selection without replacement
            idxs = np.random.choice(
                len(self.processed_data), 
                self.batch_size, 
                replace=False
            )
        
        # Create batch by stacking the selected items
        batch = {
            k: torch.stack([self.processed_data[i][k] for i in idxs], dim=0).to(self.device)
            for k in self.processed_data[0]
        }
        
        yield batch
                

"""----add more batch generators as needed below----"""