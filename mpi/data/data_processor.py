"""
DataProcessor: Main processor that combine all processor into a data pipeline
"""

from collections import defaultdict
from tqdm import tqdm

class DataProcessor:
    """
    Main processor that handles the completed data pipeline.
    
    single_task_pipeline
    - requires: sequence_processor

    multi_task_pipeline
    - requires: sequence_processor

    multi_task_pipeline_with_segmenter
    - requires: sequence_processor, segmenter_processor
    """
    
    def __init__(self):
        """
        Initialize the DataProcessor with available pipelines.
        """
        self.pipelines = {
            'single_task': self._single_task_pipeline,
            'multi_task': self._multi_task_pipeline,
            'multi_task_segment': self._multi_task_pipeline_with_segmenter,
            'mix_dataset': self._mix_dataset_pipeline,
            # Add more pipelines as needed
        }


    def process_dataset(self, dataset, pipeline_name, processors):
        """
        Process a dataset into sequences.
        
        Args:
            dataset: A TrajectoryDataset
            pipeline_name: Name of the pipeline to use
            processors: Dict of processors to pass to the pipeline
            example: {'segmenter_processor': ..., 'sequence_processor': ...}
            
        Returns:
            Dict mapping task_id to lists of processed sequences
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found. These are availble pipelines: {list(self.pipelines.keys())}")
        
        # Get the pipeline function
        pipeline = self.pipelines[pipeline_name]

        # Call pipeline with unpacked processors
        return pipeline(dataset, **processors)


    def _single_task_pipeline(self, dataset, sequence_processor):
        """
        Process a dataset into sequences without task-specific organization.
        
        Args:
            dataset: A TrajectoryDataset
            
        Returns:
            List of processed sequences
        """

        sequences = []
        trajecotires = dataset[0].get_trajectories()
        
        for trajectory in tqdm(trajecotires, desc="Processing episodes"):
            # Process each trajectory into sequences
            processed_sequences = sequence_processor.process_episode(trajectory)
            sequences.extend(processed_sequences)
        
        return sequences


    def _multi_task_pipeline(self, dataset, sequence_processor):
        """
        Process a dataset into "task-specific" sequences.
        
        Args:
            dataset: A TrajectoryDataset
            
        Returns:
            Dict mapping task_id to lists of processed sequences
        """

        task_datasets = defaultdict(list)
        task_counts = defaultdict(int)

        trajecotires = dataset.get_trajectories()

        # If dataset is already organized by task, 
        # process each task separately
        for task_id, trajectoies in tqdm(trajecotires.items(), desc="Processing trajectories"):
            task_counts[task_id] += len(trajectoies)
            for trajectory in trajectoies:
                sequences = sequence_processor.process_episode(trajectory)
                task_datasets[task_id].extend(sequences)
    
        # Print task distribution
        print("\nTask distribution:")
        for task_id, count in task_counts.items():
            print(f" {task_id}: {count} segments, {len(task_datasets[task_id])} sequences")
        
        return task_datasets


    def _multi_task_pipeline_with_segmenter(self, dataset, sequence_processor, segmenter_processor):
        """
        Process a dataset into "task-specific" sequences.
        
        Args:
            dataset: A TrajectoryDataset
            
        Returns:
            Dict mapping task_id to lists of processed sequences
        """

        task_datasets = defaultdict(list)
        task_counts = defaultdict(int)

        trajecotires = dataset.get_trajectories()

        # If dataset is already organized by task, 
        # process each task separately
        for trajectory in tqdm(trajecotires, desc="Processing trajectories"):
            
            # Segment trajectory into task-specific segments
            segments = segmenter_processor.segment_trajectory(trajectory)
            
            # Process segments into sequences
            for segment in segments:
                task_id = segment["task_id"]
                task_counts[task_id] += 1
                
                # Process segment into fixed-length sequences
                sequences = sequence_processor.process_episode(segment)
                
                # Add sequences to task-specific dataset
                task_datasets[task_id].extend(sequences)
    
        # Print task distribution
        print("\nTask distribution:")
        for task_id, count in task_counts.items():
            print(f" {task_id}: {count} segments, {len(task_datasets[task_id])} sequences")
        
        return task_datasets


    def _mix_dataset_pipeline(self, datasets: list, sequence_processor):
        """
        Process multiple datasets into sequences.
        
        Args:
            datasets: List of TrajectoryDataset objects
            
        Returns:
            List of processed sequences
        """
        sequences = []
        
        for dataset in tqdm(datasets, desc="Processing datasets"):
            trajecotires = dataset.get_trajectories()
            for trajectory in trajecotires:
                processed_sequences = sequence_processor.process_episode(trajectory)
                sequences.extend(processed_sequences)
        
        return sequences

    '''add more pipeline below as needed'''
