Minari Dataset segment task dataflow graph:

```mermaid
graph LR
    %% Load and Start
    A1["Minari Dataset"] --> B1["split_task()"]
    
    %% Split into subtrajectories
    B1 --> C1["segment_trajectory_by_subtasks()"]
    C1 --> C2["4 task trajectories per episode"]

    %% Sliding window or padding
    C2 --> D1["process_episode()"]
    D1 --> D2{"Length > Horizon *H*?"}
    D2 -- Yes --> D3["Split the trajectory with sliding window"]
    D2 -- No --> D4["Pad with Zeros to *H*"]

    %% Processed sequences
    D3 --> E1["processed sequence"]
    D4 --> E1

    %% Group by task
    E1 --> F1["Group into 4 task"]

    F1 --> G1["get_batches"]

    %% Batch creation
    G1 --> H1["Create batch within each task"]

    %% Model Input
    H1 --> I1["Model input (Batch)"]

    %% Styling
    classDef main fill:#e6fffa,stroke:#333,stroke-width:2px
    classDef decision fill:#fff0e6,stroke:#333,stroke-width:1px
    classDef output fill:#ffffe6,stroke:#333,stroke-width:2px
    classDef function fill:#ffe6fa,stroke:#333,stroke-width:1px

    class A1 main
    class B1,C1,D1,G1 function
    class I1 output
    class D2 decision
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
        achieved --> ag_2["Task 2: array[T, ag_dim_2]"]
        achieved --> ag_3["Task 3: array[T, ag_dim_3]"]
        achieved --> ag_4["Task 4: array[T, ag_dim_4]"]

        desired --> dg_1["Task 1: array[T, dg_dim_1]"]
        desired --> dg_2["Task 2: array[T, dg_dim_2]"]
        desired --> dg_3["Task 3: array[T, dg_dim_3]"]
        desired --> dg_4["Task 4: array[T, dg_dim_4]"]

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

        task1 --> t1_traj1["[Segmented Trajectory 1, ...]"]

        task2 --> t2_traj1["[Segmented Trajectory 1, ...]"]

        task3 --> t3_traj1["[Segmented Trajectory 1, ...]"]

        task4 --> t4_traj1["[Segmented Trajectory 1, ...]"]
    end
```

After we have the above task dataset schema, we will processe the sequences using sliding window or padding, the data shape for each sequence is:
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
