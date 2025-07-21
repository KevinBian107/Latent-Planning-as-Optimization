import torch
import torch.nn as nn
from agent.src.layers.block import Block
from agent.src.models.conditional_decision_transformer import ConditionalDecisionTransformer

"""
posterior distribution
"""

class DecisionTransformerEncoder(nn.Module):

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.attention_pool = TrajectoryAttentionPool

        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )
        self.apply(self._init_weights)


    def forward(self, timesteps, states, actions, returns_to_go):

        B, T, _ = states.shape
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)

        h = self.embed_ln(h)
        h = self.transformer(h)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        h_seq = h.reshape(B, 3 * T, self.h_dim)  # [B, L, H]

        return h_seq,h


class DTdecoder(nn.Module):
    def __init__(self,h_dim,state_dim,act_dim,use_action_tanh = True):
        super().__init__()
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else [])))
    def forward(self,h):

        return_preds = self.predict_rtg(h[:,2])     # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:,2])    # predict next state given r, s, a
        action_preds = self.predict_action(h[:,1]) 
        return action_preds,state_preds,return_preds


class TrajectoryAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # [1, 1, H]
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

    def forward(self, h_seq):  # h_seq: [B, L, H]
        B = h_seq.size(0)
        q = self.query.expand(B, -1, -1)  # [B, 1, H]
        pooled, _ = self.attn(q, h_seq, h_seq)  # [B, 1, H]
        return pooled.squeeze(1)  # [B, H]
    
class TrajectoryEncoder(nn.Module):
    def __init__(self, h_dim, z_dim, num_latent):
        super().__init__()
        self.num_latent = num_latent
        self.query = nn.Parameter(torch.randn(1, num_latent, h_dim))  # [1, N, H]

        self.reward_embed = nn.Sequential(
            nn.Linear(1, h_dim),
            nn.Tanh()
        )

        self.attn = nn.MultiheadAttention(h_dim, num_heads=1, batch_first=True)

        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h_seq, total_rewards):  
        """
        h_seq: [B, L, H] - transformer encoded trajectory tokens
        total_rewards: [B, 1] - scalar total reward for each trajectory
        """
        B = h_seq.size(0)

        # 1. Embed the scalar reward -> [B, 1, H]
        reward_token = self.reward_embed(total_rewards).unsqueeze(1)

        # 2. Concatenate to sequence -> [B, L+1, H]
        conditioned_seq = torch.cat([h_seq, reward_token], dim=1)

        # 3. Query-based attention pooling
        q = self.query.expand(B, -1, -1)  # [B, N, H]
        attn_out, _ = self.attn(q, conditioned_seq, conditioned_seq)  # [B, N, H]

        # 4. Project to latent space
        mu = self.fc_mu(attn_out)         # [B, N, z_dim]
        logvar = self.fc_logvar(attn_out) # [B, N, z_dim]
        z = self.reparameterize(mu, logvar)  # [B, N, z_dim]

        return z, mu, logvar
    
class PriorEncoder(nn.Module):
    def __init__(self, total_reward_dim, z_dim, h_dim):
        super().__init__()
        self.total_reward_embed = nn.Sequential(
            nn.Linear(total_reward_dim, h_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)

    def forward(self, total_rewards):  # [B, 1]
        h = self.total_reward_embed(total_rewards)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
class VAE(nn.Module):
    def __init__(self,state_dim, 
                    act_dim, 
                    n_blocks, 
                    h_dim, 
                    context_len,
                    n_heads, 
                    drop_p,
                    z_dim,
                    num_latent):
        """
        transformer get the trajectory representation
        """
        """
        encoder to get the representation z
        """
        super().__init__()
        self.transformer = DecisionTransformerEncoder(state_dim, 
                                                      act_dim, 
                                                      n_blocks, 
                                                      h_dim, 
                                                      context_len,
                                                      n_heads, 
                                                      drop_p)
        self.encoder = TrajectoryEncoder(h_dim, 
                                         z_dim, 
                                         num_latent)
        
        self.DTdecoder = DTdecoder(h_dim=h_dim,
                                   state_dim=state_dim,
                                   act_dim=act_dim,
                                   use_action_tanh=False)
        
        self.TrajectoryDecoder = ConditionalDecisionTransformer(state_dim, 
                                                                 act_dim, 
                                                                 n_blocks, 
                                                                 h_dim, 
                                                                 context_len,
                                                                 n_heads, 
                                                                 drop_p, 
                                                                 max_timestep=4096, 
                                                                 action_tanh=True)
        self.RewardDecoder = nn.Sequential(nn.Linear(h_dim,1))
        self.z_projector = nn.Sequential(nn.Linear(z_dim,h_dim))

        self.prior_encoder = PriorEncoder(total_reward_dim=1,z_dim = z_dim,h_dim=h_dim)


        
        

    def forward(self,states,actions,returns_to_go,timesteps,rewards,disable_test = False):
        h_sequence,h = self.transformer.forward(states=states,actions=actions,returns_to_go=returns_to_go,timesteps=timesteps.squeeze(-1))
        dt_pred_action,dt_pred_state,dt_pred_return_to_go = self.DTdecoder.forward(h)
        z, mu, logvar = self.encoder.forward(h_sequence,total_rewards=rewards)
        z_prior, mu_prior, logvar_prior = self.prior_encoder(total_rewards=rewards)
        
        # if disable_test:
        #     z = torch.randn_like(z)
        z = self.z_projector(z)
        pred_action, pred_state = self.TrajectoryDecoder.forward(timesteps = timesteps.squeeze(-1), 
                                        states = states, 
                                        actions = actions, 
                                        z_latent=z)
        if disable_test:
            pred_action, pred_state = self.TrajectoryDecoder.forward(timesteps = timesteps.squeeze(-1), 
                                            states = states, 
                                            actions = actions, 
                                            z_latent=torch.rand_like(z))
        pred_reward = self.RewardDecoder.forward(z)

        return (pred_action,pred_state,pred_reward),(z,mu,logvar),(z_prior, mu_prior, logvar_prior)
    
    def generate(self,rewards,actions,states,timesteps):
        z,mu,logvar = self.prior_encoder.forward(total_rewards=rewards)
        z = self.z_projector(z).unsqueeze(1)
        pred_action, pred_state = self.TrajectoryDecoder.forward(timesteps = timesteps.squeeze(-1), 
                                        states = states, 
                                        actions = actions, 
                                        z_latent=z)
        return pred_action,pred_state
    

    




    

