import torch
import torch.nn as nn
from agent.src.layers.block import CrossAttnBlock,RotaryPositionalEmbeddings

class ConditionalDecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, action_tanh=True):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.context_len = context_len

        # Input Embeddings
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_action = nn.Linear(act_dim, h_dim)

        # Time embedding for states and actions
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_ln = nn.LayerNorm(h_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttnBlock(h_dim, 2 * context_len, n_heads, drop_p)
            for _ in range(n_blocks)
        ])

        # Output heads
        self.predict_action = nn.Sequential(
            nn.Linear(h_dim, act_dim),
            nn.Tanh() if action_tanh else nn.Identity()
        )
        self.predict_state = nn.Linear(h_dim, state_dim)

    def forward(self, timesteps, states, actions, z_latent):
        B, T, _ = states.shape

        # 1. Embed states and actions separately, add time embeddings
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # 2. Interleave state and action embeddings: (s0, a0, s1, a1, ..., sT)
        seq = torch.stack((state_embeddings, action_embeddings), dim=2)
        seq = seq.reshape(B, 2 * T, self.h_dim)
        seq = self.embed_ln(seq)

        # 3. Pass through transformer blocks
        h = seq
        for block in self.blocks:
            h = block((h, z_latent))  # (token embeddings, cross attention memory)

        # 4. Predict next action and next state based on last tokens
        # last action embedding => predict next action
        #last_state_emb = h[:, -2]  # last state token
        #last_action_emb = h[:, -1]  # last action token

        state_emb = h[:, 0::2, :]
        action_emb = h[:, 1::2, :]

        pred_action = self.predict_action(state_emb)
        pred_state = self.predict_state(action_emb)

        return pred_action, pred_state
