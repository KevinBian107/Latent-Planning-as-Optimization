# lpt_maml_train.py  ── replaces the old LPT training script
import os, sys, torch, numpy as np
from collections import defaultdict
from tqdm import tqdm
import minari
import matplotlib.pyplot as plt

# local import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.LPT import LatentPlannerModel

# ───────────────────────── device ─────────────────────────
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ───────────────────────── params ─────────────────────────
MAX_LEN, HIDDEN_SIZE, N_LAYER, N_HEAD = 15, 2, 3, 1
BATCH_SIZE = 16                     # windows in inner‑loop batches
NUM_META_ITERS = 2000               # ── MAML: outer steps
META_BATCH_SIZE = 4                 # ── MAML: tasks per outer step
K_INNER = 3                         # ── MAML: gradient steps
Q_QUERY = 10                        # ── MAML: query windows
INNER_LR = 1e-3                     # ── MAML
OUTER_LR = 1e-4
LATENT_DIM = 8                      # if you sample z0; adjust to your model
WINDOW_SIZE = 30                    # plotting
SMOOTH_ALPHA = 0.05

context_len = MAX_LEN   # for model ctor

# ───────────────────── kitchen indices & thresholds ───────────────
MICROWAVE_IDX = 31
KETTLE_IDX_X, KETTLE_IDX_Y, KETTLE_IDX_Z = 32, 33, 34
LIGHT_SWITCH_IDX, SLIDE_CABINET_IDX = 26, 28
MICROWAVE_THRESHOLD, KETTLE_MOVE_THRESHOLD = 0.2, 0.1
LIGHT_SWITCH_THRESHOLD, SLIDE_CABINET_THRESHOLD = -0.6, 0.2

# ───────────────────── helper functions ───────────────────────────
def detect_subtasks(ep):
    obs = ep.observations["observation"]
    init, final = obs[0], obs[-1]
    subs = []
    if final[MICROWAVE_IDX] > MICROWAVE_THRESHOLD: subs.append("microwave")
    if (np.linalg.norm(final[KETTLE_IDX_X:KETTLE_IDX_Z+1] -
                       init[KETTLE_IDX_X:KETTLE_IDX_Z+1]) > KETTLE_MOVE_THRESHOLD):
        subs.append("kettle")
    if final[LIGHT_SWITCH_IDX] < LIGHT_SWITCH_THRESHOLD: subs.append("light")
    if final[SLIDE_CABINET_IDX] > SLIDE_CABINET_THRESHOLD: subs.append("slidecabinet")
    return subs

def determine_task_id(ep):
    subs = detect_subtasks(ep)
    if not subs: return 0
    return hash("_".join(sorted(subs))) % 5

def process_episode(ep, max_len=MAX_LEN):
    obs = torch.tensor(ep.observations["observation"][:-1], dtype=torch.float32, device=device)
    act = torch.tensor(ep.actions, dtype=torch.float32, device=device)
    rew = torch.tensor(ep.rewards, dtype=torch.float32, device=device)
    done = torch.tensor(ep.terminations, dtype=torch.bool, device=device)
    rtg = rew.flip(0).cumsum(0).flip(0).unsqueeze(-1)
    prev_act = torch.cat([torch.zeros_like(act[:1]), act[:-1]], 0)
    timesteps = torch.arange(len(obs), device=device).unsqueeze(-1)

    if len(obs) < max_len: return []
    seqs = []
    for i in range(len(obs)-max_len+1):
        seqs.append({
            "observations": obs[i:i+max_len],
            "actions":      act[i:i+max_len],
            "reward":       rew[i:i+max_len].unsqueeze(-1),
            "done":         done[i:i+max_len].unsqueeze(-1),
            "return_to_go": rtg[i:i+max_len],
            "prev_actions": prev_act[i:i+max_len],
            "timesteps":    timesteps[i:i+max_len]
        })
    return seqs

def organize_data_by_task(dataset):
    task_sets, counts = defaultdict(list), defaultdict(int)
    for ep in tqdm(dataset, desc="organising"):
        tid = determine_task_id(ep)
        counts[tid]+=1
        task_sets[tid].extend(process_episode(ep))
    for t,d in task_sets.items():
        print(f"task {t}: {len(d)} windows, {counts[t]} episodes")
    return task_sets

def get_batch(seq_list, size):
    idx = np.random.choice(len(seq_list), size=size, replace=len(seq_list)<size)
    batch = [seq_list[i] for i in idx]
    return {k: torch.stack([b[k] for b in batch]).to(device) for k in batch[0]}

# ─────────────────────────── MAML utils ────────────────────────────
def split_support_query(seq_list, k, q):
    idx = np.random.permutation(len(seq_list))
    if len(idx) < k+q: idx = np.resize(idx, k+q)
    sup, qry = idx[:k], idx[k:k+q]
    return [seq_list[i] for i in sup], [seq_list[i] for i in qry]

from torch.nn.utils.stateless import functional_call  # PyTorch ≥2.0

def forward_with_params(model, params, **kwargs):
    return functional_call(model, params, kwargs)

# ─────────────────────────── training ──────────────────────────────
def main():
    ds = minari.load_dataset('D4RL/kitchen/mixed-v2', download=True)
    task_data = organize_data_by_task(ds)

    state_dim = ds[0].observations["observation"].shape[1]
    act_dim   = ds[0].actions.shape[1]

    model = LatentPlannerModel(state_dim, act_dim, HIDDEN_SIZE,
                               context_len, N_LAYER, N_HEAD, device).to(device)
    
    outer_opt = torch.optim.Adam(model.parameters(), lr=OUTER_LR)


    meta_total_losses, meta_a_losses, meta_r_losses = [], [], []
    task_losses = {tid: [] for tid in task_data}

    for it in range(NUM_META_ITERS):
        outer_opt.zero_grad()
        outer_total, outer_a, outer_r = 0.0, 0.0, 0.0

        sel_tasks = np.random.choice(list(task_data.keys()),
                                     min(META_BATCH_SIZE, len(task_data)),
                                     replace=False)

        for tid in sel_tasks:
            support, query = split_support_query(
                task_data[tid], K_INNER*BATCH_SIZE, Q_QUERY)

            fast = {n: p.clone() for n, p in model.named_parameters()}

            # ---- inner loop ----
            for _ in range(K_INNER):
                b = get_batch(support, BATCH_SIZE)
                pa, ps, pr = forward_with_params(
                    model, fast,
                    states=b["observations"],
                    actions=b["prev_actions"],
                    timesteps=b["timesteps"].squeeze(-1),
                    rewards=b["reward"],
                    batch_inds=torch.arange(BATCH_SIZE, device=device))
                la = torch.nn.functional.mse_loss(pa, b["actions"][:,-1])
                lr = torch.nn.functional.mse_loss(pr, b["reward"][:,-1,0])
                grads = torch.autograd.grad(la+lr, fast.values(), create_graph=True)
                fast = {k: w - INNER_LR*g for (k,w),g in zip(fast.items(), grads)}

            # ---- outer query loss ----
            qb = get_batch(query, Q_QUERY)
            pa, ps, pr = forward_with_params(
                model, fast,
                states=qb["observations"],
                actions=qb["prev_actions"],
                timesteps=qb["timesteps"].squeeze(-1),
                rewards=qb["reward"],
                batch_inds=torch.arange(Q_QUERY, device=device))
            la_q = torch.nn.functional.mse_loss(pa, qb["actions"][:,-1])
            lr_q = torch.nn.functional.mse_loss(pr, qb["reward"][:,-1,0])
            outer_task_loss = la_q + lr_q

            outer_total += outer_task_loss
            outer_a     += la_q
            outer_r     += lr_q
            task_losses[tid].append(outer_task_loss.item())

        outer_total /= len(sel_tasks); outer_a /= len(sel_tasks); outer_r /= len(sel_tasks)
        outer_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        outer_opt.step()

        # ---- store for plots ----
        meta_total_losses.append(outer_total.item())
        meta_a_losses.append(outer_a.item())
        meta_r_losses.append(outer_r.item())

        if (it+1) % 50 == 0:
            print(f"[{it+1:4d}/{NUM_META_ITERS}] "
                  f"total {outer_total.item():.4f} | "
                  f"a {outer_a.item():.4f} | r {outer_r.item():.4f}")

    # save model
    os.makedirs("results/weights", exist_ok=True)
    torch.save(model.state_dict(), "results/weights/lpt_maml.pt")

    # ═════════════════ plotting (unchanged layouts) ════════════════════
    plt.figure(figsize=(18,12))

    # 1 Raw losses
    plt.subplot(2,3,1)
    plt.plot(meta_total_losses, alpha=.3, label="Total loss (raw)")
    plt.plot(meta_a_losses,     alpha=.3, label="Action loss (raw)")
    plt.plot(meta_r_losses,     alpha=.3, label="Reward loss (raw)")
    plt.xlabel("Outer Step"); plt.ylabel("Loss"); plt.title("Raw Training Losses")
    plt.legend(); plt.grid(True)

    # 2 Moving average
    def moving_avg(x, w): return np.convolve(x, np.ones(w)/w, mode='valid')
    win = min(WINDOW_SIZE, len(meta_total_losses))
    plt.subplot(2,3,2)
    plt.plot(moving_avg(meta_total_losses, win), label="Total (MA)")
    plt.plot(moving_avg(meta_a_losses,     win), label="Action (MA)")
    plt.plot(moving_avg(meta_r_losses,     win), label="Reward (MA)")
    plt.xlabel("Outer Step"); plt.ylabel("Loss")
    plt.title(f"Moving Average (window={win})"); plt.legend(); plt.grid(True)

    # 3 Exponential moving average
    def ema(arr, a=SMOOTH_ALPHA):
        out=np.zeros_like(arr,dtype=float); out[0]=arr[0]
        for i in range(1,len(arr)): out[i]=a*arr[i]+(1-a)*out[i-1]
        return out
    plt.subplot(2,3,3)
    plt.plot(ema(meta_total_losses), label="Total (EMA)")
    plt.plot(ema(meta_a_losses),     label="Action (EMA)")
    plt.plot(ema(meta_r_losses),     label="Reward (EMA)")
    plt.xlabel("Outer Step"); plt.ylabel("Loss")
    plt.title(f"Exponential Moving Average (α={SMOOTH_ALPHA})")
    plt.legend(); plt.grid(True)

    # 4 Per‑task EMA curves
    plt.subplot(2,3,4)
    for tid, lst in task_losses.items():
        if lst: plt.plot(ema(lst), label=f"Task {tid}")
    plt.xlabel("Outer Step"); plt.ylabel("Loss")
    plt.title("Per‑Task Loss (EMA)")
    plt.legend(); plt.grid(True)

    # 5 Loss distribution
    plt.subplot(2,3,5)
    plt.hist(meta_total_losses, bins=30, alpha=.7, label="Total")
    plt.hist(meta_a_losses,     bins=30, alpha=.5, label="Action")
    plt.hist(meta_r_losses,     bins=30, alpha=.5, label="Reward")
    plt.xlabel("Loss value"); plt.ylabel("Frequency"); plt.title("Loss distribution")
    plt.legend(); plt.grid(True)

    # 6 Action/total ratio
    plt.subplot(2,3,6)
    ratio = np.array(meta_a_losses)/(np.array(meta_total_losses)+1e-8)
    plt.plot(ema(ratio), label="Action/Total")
    plt.axhline(.5, color='r', ls='--', label="Equal contribution")
    plt.xlabel("Outer Step"); plt.ylabel("Ratio"); plt.ylim(0,1)
    plt.title("Action Loss Contribution")
    plt.legend(); plt.grid(True)

    plt.tight_layout(); plt.show()

    # -------- Bar chart comparing first/second half per task ----------
    plt.figure(figsize=(12,6))
    bar_data = {}
    for tid, lst in task_losses.items():
        if lst:
            half = len(lst)//2
            bar_data[tid] = {
                "overall": np.mean(lst),
                "first":   np.mean(lst[:half]) if half else 0,
                "second":  np.mean(lst[half:]) if half else 0
            }
    if bar_data:
        tids = list(bar_data.keys()); x = np.arange(len(tids)); w=.25
        plt.bar(x-w, [bar_data[t]['first']  for t in tids], w, label="First half")
        plt.bar(x  , [bar_data[t]['overall']for t in tids], w, label="Overall")
        plt.bar(x+w, [bar_data[t]['second'] for t in tids], w, label="Second half")
        plt.xticks(x, tids); plt.xlabel("Task ID"); plt.ylabel("Average loss")
        plt.title("Task performance comparison"); plt.legend(); plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    main()
