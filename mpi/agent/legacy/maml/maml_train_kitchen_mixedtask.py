import torch
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import minari
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import pdb

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.LPT import LatentPlannerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    device = torch.device("mps")


device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ───────────────────────────── Hyper-params ──────────────────────────────────
MAX_LEN      = 15
HIDDEN_SIZE  = 16
N_LAYER      = 3
N_HEAD       = 1
BATCH_SIZE   = 8           # used in inner loop
# ---------- MAML ----------
NUM_META_ITERS  = 1000      # outer steps
META_BATCH_SIZE = 4         # tasks per outer update
K_INNER         = 2        # inner gradient steps
Q_QUERY         = 8        # query windows
INNER_LR        = 5e-3
OUTER_LR        = 5e-4
# ---------- plotting ----------
WINDOW_SIZE  = 30
SMOOTH_ALPHA = 0.05

context_len = MAX_LEN

# ───────────────────────────── Segmentation constants ───────────────────────
TASK_KEYS = ["microwave", "kettle", "light switch", "slide cabinet"]
PROXIMITY_THRESHOLDS = {
    "microwave": 0.2,
    "kettle":    0.3,
    "light switch": 0.2,
    "slide cabinet": 0.2,
}
STABILITY_DURATION = 20

# ───────────────────────────── Trajectory segmentation ──────────────────────
def segment_trajectory_by_subtasks(full_episode,
                                   task_goal_keys,
                                   proximity_thresholds,
                                   stability_duration):
    """
    Break a long Kitchen episode into sub-trajectories where each task_key
    stays within the proximity_threshold for ≥ stability_duration timesteps.
    Returns a list of segments (dicts).  Each segment has segment['task_id'].
    """
    segs = []
    obs_all   = full_episode.observations
    actions   = full_episode.actions
    rewards   = full_episode.rewards
    terms     = full_episode.terminations
    truncs    = full_episode.truncations
    T = len(actions)
    if T == 0 or stability_duration <= 0: return segs

    start = 0
    streak = {k: 0 for k in task_goal_keys}
    done_tasks = set()

    for t in range(T):
        triggered = None
        for k in task_goal_keys:
            if k in done_tasks: continue
            if k not in obs_all["achieved_goal"] or k not in obs_all["desired_goal"]:
                raise ValueError(f"key {k} missing in episode obs")

            achieved = obs_all["achieved_goal"][k][t]
            desired  = obs_all["desired_goal"][k][0]
            diff = np.linalg.norm(achieved - desired) if np.ndim(achieved) else abs(achieved - desired)
            if diff < proximity_thresholds[k]:
                streak[k] += 1
            else:
                streak[k]  = 0
            if streak[k] >= stability_duration:
                triggered = k
                break

        if triggered:
            end = t
            segs.append(_make_segment(obs_all, actions, rewards,
                                      start, end, triggered))
            done_tasks.add(triggered)
            start = end + 1
            streak = {k: 0 for k in task_goal_keys if k not in done_tasks}
            if len(done_tasks) == len(task_goal_keys): break

    # trailing part for any remaining task
    remaining = list(set(task_goal_keys) - done_tasks)
    if start < T and remaining:
        segs.append(_make_segment(obs_all, actions, rewards,
                                  start, T-1, remaining[0]))
    return segs

def _make_segment(obs_all, acts, rews, s, e, task_id):
    seg_obs = {
        "achieved_goal": {task_id: obs_all["achieved_goal"][task_id][s:e+1]},
        "desired_goal":  {task_id: obs_all["desired_goal"][task_id][s:e+1]},
        "observation":   obs_all["observation"][s:e+1],
    }
    seg = {
        "observations": seg_obs,
        "actions":      acts[s:e+1],
        "rewards":      rews[s:e+1],
        "terminations": np.concatenate([np.zeros(e-s, bool), [True]]),
        "truncations":  np.zeros(e-s+1, bool),
        "task_id":      task_id,
    }
    return seg

def split_task(dataset):
    seg_ds = defaultdict(list)
    task_counts = defaultdict(int)
    for i, traj in enumerate(dataset):
        for seg in segment_trajectory_by_subtasks(
                traj, TASK_KEYS, PROXIMITY_THRESHOLDS, STABILITY_DURATION):
            seqs = process_episode(seg)            # sliding windows
            seg_ds[seg["task_id"]].extend(seqs)
            task_counts[seg["task_id"]] += len(seqs)
        print("processed trajectory", i+1, "of", len(dataset))
    for k, v in task_counts.items():
        print(f"Task {k}: {v} sequences")
    return seg_ds

# ───────────────────────────── Windowing ─────────────────────────────────────
def process_episode(ep, max_len=MAX_LEN):
    obs  = torch.tensor(ep["observations"]["observation"][:-1], dtype=torch.float32)
    #desired_goal = torch.tensor(list(ep["observations"]["desired_goal"].values())[0][:-1], dtype=torch.float32)
    #achieved_goal = torch.tensor(list(ep["observations"]["achieved_goal"].values())[0][:-1], dtype=torch.float32)
    #obs = torch.cat([pre_obs,desired_goal,achieved_goal],dim = -1)



    acts = torch.tensor(ep["actions"], dtype=torch.float32)
    rews = torch.tensor(ep["rewards"], dtype=torch.float32)
    rtg  = rews.flip(0).cumsum(0).flip(0).unsqueeze(-1)
    prev = torch.cat([torch.zeros_like(acts[:1]), acts[:-1]], 0)
    ts   = torch.arange(len(obs)).unsqueeze(-1)

    seqs = []
    if len(obs) < max_len:   # pad short episode to one window
        pad = max_len - len(obs)
        obs  = torch.cat([obs,  torch.zeros(pad, obs.shape[1])], 0)
        acts = torch.cat([acts, torch.zeros(pad, acts.shape[1])], 0)
        rews = torch.cat([rews, torch.zeros(pad)], 0)
        rtg  = torch.cat([rtg,  torch.zeros(pad, 1)], 0)
        prev = torch.cat([prev, torch.zeros(pad, prev.shape[1])], 0)
        ts   = torch.cat([ts,   torch.zeros(pad, 1)], 0)
        seqs.append({
            "observations": obs, "actions": acts, "reward": rews.unsqueeze(-1),
            "return_to_go": rtg, "prev_actions": prev, "timesteps": ts,
        })
        return seqs

    for i in range(len(obs) - max_len + 1):
        seqs.append({
            "observations": obs[i:i+max_len],
            "actions":      acts[i:i+max_len],
            "reward":       rews[i:i+max_len].unsqueeze(-1),
            "return_to_go": rtg[i:i+max_len],
            "prev_actions": prev[i:i+max_len],
            "timesteps":    ts[i:i+max_len],
        })
    return seqs

# ───────────────────────────── Batch helpers ─────────────────────────────────
def get_batch(seq_list, size):
    idx = np.random.choice(len(seq_list), size=size, replace=len(seq_list) < size)
    batch = [seq_list[i] for i in idx]
    return {k: torch.stack([b[k] for b in batch]).to(device) for k in batch[0]}

# MAML utilities
def split_support_query(seq_list, k, q):
    idx = np.random.permutation(len(seq_list))
    if len(idx) < k + q: idx = np.resize(idx, k + q)
    return [seq_list[i] for i in idx[:k]], [seq_list[i] for i in idx[k:k+q]]

from torch.nn.utils.stateless import functional_call
def fwd_with_params(model, params, **batch_kwargs):
    """
    Stateless forward pass that uses `params` as the model’s weights.
    Accepts exactly the same keyword args that `LatentPlannerModel.forward` expects.
    """
    return functional_call(model, params, (), batch_kwargs)

# ───────────────────────────── EMA / Moving avg ─────────────────────────────
def moving_avg(x, w): return np.convolve(x, np.ones(w)/w, mode="valid")
def ema(x, a=SMOOTH_ALPHA):
    out = np.zeros_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)): out[i] = a * x[i] + (1-a) * out[i-1]
    return out

# ───────────────────────────── Main Train ───────────────────────────────────
def main():
    print("loading Minari dataset …")
    raw_ds = minari.load_dataset("D4RL/kitchen/complete-v2", download=True)
    task_ds = split_task(raw_ds)                               # segmentation

    state_dim = raw_ds[0].observations["observation"].shape[1] 
    act_dim   = raw_ds[0].actions.shape[1]

    model = LatentPlannerModel(
        state_dim=state_dim,
        act_dim=act_dim,
        h_dim=HIDDEN_SIZE,
        context_len=context_len,
        n_blocks=N_LAYER,
        n_heads=N_HEAD,
        device=device,
    ).to(device)

    outer_opt = torch.optim.Adam(model.parameters(), lr=OUTER_LR)

    # logs
    meta_total, meta_a, meta_r = [], [], []
    task_losses = {tid: [] for tid in task_ds}

    # =============== MAML training loop ===============
    for it in tqdm(range(NUM_META_ITERS)):
        outer_opt.zero_grad()
        outer_total, outer_a, outer_r = 0.0, 0.0, 0.0

        tasks = np.random.choice(list(task_ds.keys()),
                                 min(META_BATCH_SIZE, len(task_ds)),
                                 replace=False)

        for tid in tasks:
            support, query = split_support_query(
                task_ds[tid], K_INNER * BATCH_SIZE, Q_QUERY)

            # ─── fix when we clone the fast weights ───
            fast = {n: p.clone().detach().requires_grad_(True) 
                    for n, p in model.named_parameters()}


            # ----- inner gradient steps -----
            for _ in range(K_INNER):
                b = get_batch(support, BATCH_SIZE)
                pred_action, pred_state, pred_reward = fwd_with_params(
                    model, 
                    fast,
                    states=b["observations"],
                    actions=b["prev_actions"],
                    timesteps=b["timesteps"].squeeze(-1),
                    rewards=b["reward"],
                    batch_inds=torch.arange(BATCH_SIZE, device=device),
                )
                la = torch.nn.functional.mse_loss(pred_action, b["actions"][:, -1])
                lr = torch.nn.functional.mse_loss(pred_reward, b["reward"][:, -1, 0])
                grads = torch.autograd.grad(la + lr, fast.values(),
                            create_graph=True, allow_unused=True)
                
                new_fast = {}
                for (name, w), g in zip(fast.items(), grads):     
                    if g is None:                       
                        new_fast[name] = w
                    else:
                        new_fast[name] = w - INNER_LR * g
                fast = new_fast

            # ----- outer query loss -----
            qb = get_batch(query, Q_QUERY)
            pa, ps, pr = fwd_with_params(
                model, fast,
                states=qb["observations"],
                actions=qb["prev_actions"],
                timesteps=qb["timesteps"].squeeze(-1),
                rewards=qb["reward"],
                batch_inds=torch.arange(Q_QUERY, device=device),
            )
            la_q = torch.nn.functional.mse_loss(pa, qb["actions"][:, -1])
            lr_q = torch.nn.functional.mse_loss(pr, qb["reward"][:, -1, 0])
            task_loss = la_q + lr_q

            outer_a += la_q
            outer_r += lr_q
            outer_total += task_loss
            task_losses[tid].append(task_loss.item())

        outer_total /= len(tasks)
        outer_a     /= len(tasks)
        outer_r     /= len(tasks)

        outer_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        outer_opt.step()

        # store logs
        meta_total.append(outer_total.item())
        meta_a.append(outer_a.item())
        meta_r.append(outer_r.item())

        if (it+1) % 50 == 0:
            print(f"[{it+1:4d}/{NUM_META_ITERS}] "
                  f"meta-loss {outer_total.item():.4f}  "
                  f"A {outer_a.item():.4f} | R {outer_r.item():.4f}")

    # save model
    os.makedirs("results/weights", exist_ok=True)
    torch.save(model.state_dict(), "results/weights/maml_kitchen.pt")
    print("training done – saved weights to results/weights/maml_kicthen.pt")

    # =============== Visualization ===============
    if not meta_total: return
    plt.figure(figsize=(18, 12))

    # 1 raw
    plt.subplot(2,3,1)
    plt.plot(meta_total, alpha=.3, label="Total (raw)")
    plt.plot(meta_a,     alpha=.3, label="Action (raw)")
    plt.plot(meta_r,     alpha=.3, label="Reward (raw)")
    plt.title("Raw training losses"); plt.legend(); plt.grid(True)

    # 2 moving average
    plt.subplot(2,3,2)
    w = min(WINDOW_SIZE, len(meta_total))
    plt.plot(moving_avg(meta_total, w), label="Total (MA)")
    plt.plot(moving_avg(meta_a, w),     label="Action (MA)")
    plt.plot(moving_avg(meta_r, w),     label="Reward (MA)")
    plt.title(f"Moving Average w={w}"); plt.legend(); plt.grid(True)

    # 3 EMA
    plt.subplot(2,3,3)
    plt.plot(ema(np.array(meta_total)), label="Total (EMA)")
    plt.plot(ema(np.array(meta_a)),     label="Action (EMA)")
    plt.plot(ema(np.array(meta_r)),     label="Reward (EMA)")
    plt.title(f"Exponential MA α={SMOOTH_ALPHA}"); plt.legend(); plt.grid(True)

    # 4 task curves
    plt.subplot(2,3,4)
    for tid, arr in task_losses.items():
        if arr: plt.plot(ema(np.array(arr)), label=f"{tid}")
    plt.title("Per-task EMA"); plt.legend(); plt.grid(True)

    # 5 distribution
    plt.subplot(2,3,5)
    plt.hist(meta_total, bins=30, alpha=.7, label="Total")
    plt.hist(meta_a,     bins=30, alpha=.5, label="Action")
    plt.hist(meta_r,     bins=30, alpha=.5, label="Reward")
    plt.title("Loss distribution"); plt.legend(); plt.grid(True)

    # 6 action / total ratio
    plt.subplot(2,3,6)
    ratio = np.array(meta_a)/(np.array(meta_total)+1e-8)
    plt.plot(ema(ratio), label="Action/Total")
    plt.axhline(.5, color="r", ls="--")
    plt.title("Action loss contribution"); plt.ylim(0,1); plt.legend(); plt.grid(True)

    plt.tight_layout(); plt.show()

    # bar chart
    plt.figure(figsize=(12,6))
    ids, first, second, overall = [], [], [], []
    for tid, arr in task_losses.items():
        if not arr: continue
        half = len(arr)//2
        ids.append(tid)
        overall.append(np.mean(arr))
        first.append(np.mean(arr[:half]) if half else 0)
        second.append(np.mean(arr[half:]) if half else 0)
    if ids:
        x = np.arange(len(ids)); w=.25
        plt.bar(x-w, first,  w, label="1st half")
        plt.bar(x,   overall,w, label="Overall")
        plt.bar(x+w, second, w, label="2nd half")
        plt.xticks(x, ids); plt.legend(); plt.grid(axis="y")
        plt.title("Task performance comparison"); plt.ylabel("Avg loss")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
