import torch
import numpy as np
from collections import deque
from abc import ABC, abstractmethod
from utils.process_obs import process_observation, kitchen_goal_obs_dict
import minari

# --------------------- 环境注册 ---------------------
ENV_REGISTRY = {
    "kitchen-mixed-v2": {
        "env_name": "D4RL/kitchen/mixed-v2",
        "step_mapping": lambda out: (process_observation(kitchen_goal_obs_dict, out[0]["observation"]), out[1], out[2], out[3], out[4]),
        "reset_mapping": lambda out: process_observation(kitchen_goal_obs_dict, out[0]["observation"]),
    },
    "kitchen-complete-v2": {
        "env_name": "D4RL/kitchen/complete-v2",
        "step_mapping": lambda out: (process_observation(kitchen_goal_obs_dict, out[0]["observation"]), out[1], out[2], out[3], out[4]),
        "reset_mapping": lambda out: process_observation(kitchen_goal_obs_dict, out[0]["observation"]),
    },
    "halfcheetah-expert-v0": {
        "env_name": "mujoco/halfcheetah/expert-v0",
        "step_mapping": lambda out: (out[0], out[1], out[2], out[3], out[4]),
        "reset_mapping": lambda out: out[0],
    }
}
# ---------------------------------------------------

class BaseInferencer(ABC):
    def __init__(self, args):
        self.args = args
        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        # 获取环境配置
        env_key = args.environment["name"]
        if env_key not in ENV_REGISTRY:
            raise ValueError(f"[Inferencer] Unknown environment: {env_key}")
        env_info = ENV_REGISTRY[env_key]
        self.env_name = env_info["env_name"]
        self.step_mapping = env_info["step_mapping"]
        self.reset_mapping = env_info["reset_mapping"]

        # 加载环境
        dataset = minari.load_dataset(self.env_name, download=True)
        self.env = dataset.recover_environment(render_mode="human")

        # 加载模型
        self.model = self._load_model().to(self.device).eval()

    def _load_model(self):
        model_name = self.args.model_name
        path = self.args.path["weights_path"]
        env_key = self.args.environment["name"].split("-")[0]

        fname = None
        if model_name == "BasicLPT":
            fname = f"lpt_{env_key}.pt"
        elif model_name == "BasicDT":
            fname = f"dt_{env_key}.pt"
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        return torch.load(f"{path}/{fname}",weights_only=False)

    def construct_obs(self, raw_obs):
        # 默认使用 reset_mapping 函数，也支持被子类 override
        return raw_obs  # 可以被子类重写或动态替换为 reset_mapping

    @abstractmethod
    def inference(self, steps=None):
        pass


class LPTInferencer(BaseInferencer):
    def inference(self, steps=1000):
        context_len = self.args.environment["context_len"]


        obs = self.reset_mapping(self.env.reset())
        state_dim = obs.shape[0]
        act_dim = self.model.act_dim

        # 初始化 context buffers
        state_buffer = deque([np.zeros(state_dim)] * (context_len - 1) + [obs], maxlen=context_len)
        action_buffer = deque([np.zeros(act_dim)] * context_len, maxlen=context_len)
        timestep_buffer = deque(range(context_len), maxlen=context_len)

        t, total_reward = 0, 0.0
        done = False

        while not done and (steps is None or t < steps):
            states = torch.tensor([list(state_buffer)], dtype=torch.float32).to(self.device)
            actions = torch.tensor([list(action_buffer)], dtype=torch.float32).to(self.device)
            rewards = torch.tensor([[300.0]], dtype=torch.float32).to(self.device)  # dummy
            timesteps = torch.tensor([list(timestep_buffer)], dtype=torch.long).to(self.device)

            pred_action, _, _ = self.model(
                states, actions, timesteps, rewards,
                batch_inds=torch.tensor([0], device=self.device)
            )
            action = pred_action.squeeze().detach().cpu().numpy()

            # Step 环境并处理返回
            raw_step = self.env.step(action)
            next_obs, reward, done, truncated, info = self.step_mapping(raw_step)
            self.env.render()

            # 更新 context
            state_buffer.append(next_obs)
            action_buffer.append(action)
            timestep_buffer.append(t + context_len)

            total_reward += reward
            t += 1
            if done or truncated:
                break;


        print("=" * 40)
        print(f"Inference finished after {t} steps.")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per step: {total_reward / t:.2f}" if t > 0 else "No steps taken.")
        print("=" * 40)



class DTInferencer(BaseInferencer):
    def inference(self, steps=1000):
        context_len = self.args.environment["context_len"]
        return_to_go = self.args.BasicDT.get("init_rtg", 154)

        raw_obs = self.env.reset()
        obs = self.reset_mapping(raw_obs)
        state_dim = obs.shape[0]
        act_dim = self.env.action_space.shape[0]

        # 初始化缓存
        states = deque([], maxlen=context_len)
        actions = deque([], maxlen=context_len)
        rewards = deque([], maxlen=context_len)
        timesteps = deque([], maxlen=context_len)
        rtgs = deque([], maxlen=context_len)

        total_reward = 0.0
        t = 0
        done = False

        def pad_tensor(seq, size, dim):
            pad_len = size - len(seq)
            if pad_len <= 0:
                return torch.stack(list(seq)[-size:])
            else:
                stacked = torch.stack(list(seq))
                if stacked.ndim == 1:
                    stacked = stacked.unsqueeze(-1)
                return torch.cat([
                    torch.zeros((pad_len, dim), dtype=torch.float32),
                    stacked
                ], dim=0)

        def pad_timestep(seq, size):
            pad_len = size - len(seq)
            if pad_len <= 0:
                return torch.tensor(list(seq)[-size:], dtype=torch.long)
            else:
                return torch.cat([
                    torch.zeros(pad_len, dtype=torch.long),
                    torch.tensor(list(seq), dtype=torch.long)
                ])

        while not done and (steps is None or t < steps):
            states.append(torch.tensor(obs, dtype=torch.float32))
            actions.append(torch.zeros(act_dim))  # dummy for prediction
            rewards.append(0.0)
            timesteps.append(t)
            rtgs.append(torch.tensor([return_to_go - total_reward], dtype=torch.float32))

            # 构造 batch
            state_batch = pad_tensor(states, context_len, state_dim).unsqueeze(0).to(self.device)
            action_batch = pad_tensor(actions, context_len, act_dim).unsqueeze(0).to(self.device)
            rtg_batch = pad_tensor(rtgs, context_len, 1).unsqueeze(0).to(self.device)
            timestep_batch = pad_timestep(timesteps, context_len).unsqueeze(0).to(self.device)

            # 模型推理
            with torch.no_grad():
                output = self.model(
                    timesteps=timestep_batch,
                    states=state_batch,
                    actions=action_batch,
                    returns_to_go=rtg_batch
                )
                action_preds = output[1]
                action = action_preds[0, -1].cpu().numpy()

            # 环境 step
            raw_step = self.env.step(action)
            next_obs, reward, done, truncated, info = self.step_mapping(raw_step)
            self.env.render()

            obs = next_obs
            total_reward += reward

            actions[-1] = torch.tensor(action, dtype=torch.float32)
            rewards[-1] = reward
            t += 1

            if done or truncated:
                break

        print("=" * 40)
        print(f"Inference finished after {t} steps.")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Average reward per step: {total_reward / t:.2f}" if t > 0 else "No steps taken.")
        print("=" * 40)