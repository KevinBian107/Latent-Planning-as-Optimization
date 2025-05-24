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
                  for k in batch[0] if k != 'task_id'}
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
                      for k in batch[0] if k != 'task_id'}


"""----add more batch generators as needed below----"""

