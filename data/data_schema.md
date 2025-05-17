Dataflow:

```mermaid
flowchart TD
    %% High level flow
    RawData[Raw Data downloaded from any source] --> DatasetAdapter
    DatasetAdapter --> DataProcessor
    DataProcessor --> ProcessedData
    ProcessedData --> BatchGenerator
    BatchGenerator --> TrainingBatches[Ready for model input]

    %% Key components expanded
    subgraph DatasetAdapter[TrajectoryDataset]
        DA_Custom["Your Custom preprocessing function to adapt to pipeline"]
    end
    
    subgraph DataProcessor[Data Processing Pipeline]
        DP_Process["Create your custom processing pipeline function"]
        DP_Registry["Register your pipeline with pipline name"]
    end

    subgraph Processors[Processor Components]
        direction TB
        PC_Base["BaseProcessor Interface <br> process()"]
        PC_Base -.->|implement| PC_Segment["Segmentation Processor"]
        PC_Base -.->|implement| PC_Sequence["Sequence Processor"]
        PC_Base -.->|implement| PC_Custom["Your Custom Processors"]
    end
    
    Processors -->|pass in| DataProcessor
    
    subgraph ProcessedData[Processed Data Structure]
        PD_Dict["Dict<br>[task_id, List[sequences]]"]
        PD_Sequence["Sequence = <br>{observations: tensor,<br>actions: tensor,<br>reward: tensor,<br>return_to_go: tensor,<br>prev_actions: tensor,<br>timesteps: tensor}"]
        PD_Dict --> PD_Sequence
    end
    
    subgraph BatchGenerator[Batch Generation]
        BG_Create["Creates task-specific batches"]
        BG_Iterate["Yields batches for model training"]
    end
    
    %% Extension points
    ExtensionPoints1["Your Task Split Logic"] -.->|replace| PC_Segment
    ExtensionPoints2["Your Process Sequence Logic"] -.->|replace| PC_Sequence
    
    classDef interface fill:#f9f,stroke:#333,stroke-width:1px
    classDef component fill:#bbdefb,stroke:#333,stroke-width:1px
    classDef dataStructure fill:#c8e6c9,stroke:#333,stroke-width:1px
    classDef extensionPoint fill:#ffe0b2,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    
    class DA_Preprocess,PC_Base interface
    class RawData,DatasetAdapter,DataProcessor,BatchGenerator,BG_Create,BG_Iterate component
    class ProcessedData,PD_Dict,PD_Sequence dataStructure
    class ExtensionPoints1,ExtensionPoints2,DA_Custom,PC_Custom extensionPoint
```

Minari Kitchen Dataset Schema:
```mermaid
graph LR
    subgraph MinariDatasetSchema

        id["id: string"]
        observations["observations: dict"]
        actions["actions: array[T, action_dim]"]
        rewards["rewards: array[T]"]
        terminations["terminations: array[T]"]
        truncations["truncations: array[T]"]
        infos["infos: dict"]

        observations --> achieved["achieved_goal: dict"]
        observations --> desired["desired_goal: dict"]
        observations --> obs["observation: array[T+1, state_dim]"]

        achieved --> ag_1["Task 1: array[T, ag_dim_1]"]
        achieved --> ag_2["..."]

        desired --> dg_1["Task 1: array[T, dg_dim_1]"]
        desired --> dg_2["..."]

    %% Note on time and dimensions
    note1["Notes: T = timesteps <br> ag_dim = achieved_goal dimension <br> dg_dim = desired_goal dimension"]:::noteStyle

    classDef noteStyle fill:#fff8dc,color:#333,stroke:#aaa,stroke-width:1px,font-style:italic;
end
```

Minari Kitchen dataset schema after task split:
```mermaid
graph LR
    subgraph SegmentedTrajectorySchema
        ST_task_id["task_id: string"]
        ST_obs["observations: dict"] --> ST_obs_achieved_goal["achieved_goal: dict {task_id: array[T, ag_dim]}"]
        ST_obs --> ST_obs_desired_goal["desired_goal: dict {task_id: array[T, dg_dim]}"]
        ST_obs --> ST_obs_observation["observation: array[T, state_dim]"]
        ST_actions["actions: array[T, action_dim]"]
        ST_rewards["rewards: array[T]"]
        ST_terms["terminations: array[T]"]
        ST_truncs["truncations: array[T]"]
end
```

Task Dataset schema:
```mermaid
graph LR
    subgraph TaskDatasetSchema
        root["task_datasets: dict"]

        root --> task1["microwave: List"]
        root --> task2["kettle: List"]
        root --> task3["light switch: List"]
        root --> task4["slide cabinet: List"]

        task1 --> t1_traj1["[Trajectory 1, ...]"]

        task2 --> t2_traj1["[Trajectory 1, ...]"]

        task3 --> t3_traj1["[Trajectory 1, ...]"]

        task4 --> t4_traj1["[Trajectory 1, ...]"]
    end
```

After we have the above task dataset schema, we will processe the trajectory using sliding window or padding, the data shape for each sequence is:
```mermaid
graph LR
    subgraph ProcessedSequenceSchema
        PS_obs["observations: tensor[*H*, state_dim]"]
        PS_actions["actions: tensor[*H*, action_dim]"]
        PS_reward["reward: tensor[*H*, 1]"]
        PS_rtg["return_to_go: tensor[*H*, 1]"]
        PS_prev_actions["prev_actions: tensor[*H*, action_dim]"]
        PS_timesteps["timesteps: tensor[*H*, 1]"]
end
```

The data shape for each batch:
```mermaid
graph LR
    subgraph BatchSchema
        B_obs["observations: tensor[BATCH_SIZE, *H*, state_dim]"]
        B_actions["actions: tensor[BATCH_SIZE, *H*, action_dim]"]
        B_reward["reward: tensor[BATCH_SIZE, *H*, 1]"]
        B_rtg["return_to_go: tensor[BATCH_SIZE, *H*, 1]"]
        B_prev_actions["prev_actions: tensor[BATCH_SIZE, *H*, action_dim]"]
        B_timesteps["timesteps: tensor[BATCH_SIZE, *H*, 1]"]
end
```
