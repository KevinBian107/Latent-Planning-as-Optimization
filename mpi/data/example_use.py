from processors import (
    SequenceProcessor,
    KitchenSegmenter,
)
from data_processor import DataProcessor
from dataset import MinariTrajectoryDataset
from batch_generator import TaskBatchGenerator

def main():
    import minari
    env_name = "D4RL/kitchen/complete-v2"
    device = "cpu"
    context_len = 50
    downloaded_data = minari.load_dataset(env_name, download=True)
    
    dataset = MinariTrajectoryDataset(dataset=downloaded_data)

    sequence_processor = SequenceProcessor(
        context_len=context_len,
        device=device
    )

    kitchen_segmenter = KitchenSegmenter(
        task_goal_keys=['microwave', 'kettle', 'light switch', 'slide cabinet'],
        proximity_thresholds={
            'microwave': 0.2,       
            'kettle': 0.3,         
            'light switch': 0.2,    
            'slide cabinet': 0.2   
        },
        stability_duration=20
    )

    data_processor = DataProcessor()

    processed_data = data_processor.process_dataset(
        dataset=dataset,
        pipeline_name='multi_task_segment',
        processors={
            'sequence_processor': sequence_processor,
            'segmenter_processor': kitchen_segmenter
        }
    )

    # Print summary information in processed_data
    print("\n------------------------------------------------")
    print("------------Processed Data Summary------------")
    print("------------------------------------------------")
    print(f"Total number of processed sequences: {len(processed_data)}")
    print(f"Available tasks in processed data: {processed_data.keys()}")
    print(f"Number of sequences in microwave task: {len(processed_data['microwave'])}")
    
    first_item = processed_data['microwave'][0]    
    print(f"Observation shape: {first_item['observations'].shape}")
    print(f"Actions shape: {first_item['actions'].shape}")
    print(f"Reward shape: {first_item['reward'].shape}")
    print(f"Return to go shape: {first_item['return_to_go'].shape}")
    print(f"Previous action shape: {first_item['prev_actions'].shape}")
    print(f"Timesteps shape: {first_item['timesteps'].shape}")


    # Create a batch generator for training data
    batch_size = 32
    print("\n------------------------------------------------")
    print("------------Batch Generator Test------------")
    print("------------------------------------------------")
    
    # Initialize batch generator with the processed data
    batch_generator = TaskBatchGenerator(
        processed_data=processed_data,
        device=device,
        batch_size=batch_size
    )
    
    # Generate a batch for one of the tasks (e.g., 'microwave')
    task_name = 'microwave'
    
    for batch in batch_generator.get_batch(task_name):
        # Print batch information
        print(f"Generated batch for task: {task_name}")
        print(f"Batch observations shape: {batch['observations'].shape}")
        print(f"Batch actions shape: {batch['actions'].shape}")
        print(f"Batch return-to-go shape: {batch['return_to_go'].shape}")
        print(f"Batch timesteps shape: {batch['timesteps'].shape}")
        break


if __name__ == "__main__":
    main()