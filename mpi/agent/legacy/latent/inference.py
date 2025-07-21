import minari

dataset = minari.load_dataset('D4RL/pointmaze/umaze-v2')

env  = dataset.recover_environment({'maze_map': [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]], 'reward_type': 'sparse', 'continuing_task': True, 'reset_target': True})
for i in dataset:
    print(i)