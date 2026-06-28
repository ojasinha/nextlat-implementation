import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from dataclasses import dataclass
from Base import Block

@dataclass
class NextLatConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    # NextLat hyperparameters
    proj_factor: int = 1.0
    lambda_next_h: float = 1.0
    lambda_kl: float = 1.0
    bias: bool = False

class LatentDynamicsModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = config.n_embd * 2 # hidden states and next token embeddings
        hidden_dim = config.proj_factor * input_dim
        hidden_dim = max(128, 128 * round(hidden_dim / 128))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=config.bias),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        )
        self.norm_x = nn.LayerNorm(input_dim, bias=config.bias)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, current_hidden_states, next_token_embeds):
        # Shape of next_token_embeds is (B, T, n_embd)
        # Shape of current_hidden_states is (B, T, n_embd)
        # Input to the mlp has shape (B, T, n_embd * 2)
        x = torch.cat([current_hidden_states, next_token_embeds], dim=-1)
        x = self.norm_x(x) # to feed gaussian into the mlp
        x = self.mlp(x) # (B, T, n_embd)
        next_hidden_states = current_hidden_states + x # Residual connection

        return next_hidden_states

class NextLat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.psi = LatentDynamicsModel(config)

        self.lambda_next_h = config.lambda_next_h
        self.lambda_kl = config.lambda_kl

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight # because pytorch stores weights as (out_feature, in_features)

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, loss_mask=None, latent_mask=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the ifnal layernorm and the classifier
        x = self.transformer.ln_f(x)

        hidden_states = x # Get the hidden states for psi

        logits = self.lm_head(x) # (B, T, vocab_size)

        # Next token loss
        loss = None
        loss_next_token = None
        loss_next_h = None
        loss_kl = None
        if targets is not None:
            loss_next_token_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='none'
            )
            if loss_mask is not None:
                loss_next_token = loss_next_token_flat[loss_mask.view(-1)].mean()
            else:
                loss_next_token = loss_next_token_flat.mean()

            # Now for the latent dynamics model
            current_h = hidden_states[:, :-1] # (B, T-1, C)
            target_h = hidden_states[:, 1:] # (B, T-1, C)

            next_token_embeds = self.transformer.wte(targets[:, :-1])

            pred_h = self.psi(current_h, next_token_embeds) # (B, T-1, C)

            # Next hidden state loss
            loss_next_h_per_token = F.smooth_l1_loss(
                pred_h,
                target_h.detach(),
                reduction='none'
            ).mean(dim=-1)
            if latent_mask is not None:
                loss_next_h = loss_next_h_per_token[latent_mask].mean()
            else:
                loss_next_h = loss_next_h_per_token.mean()

            # KL divergence
            teacher_logits = logits[:, 1:].detach() # We detach to make sure no gradients flow through this
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            student_logits = F.linear(pred_h, self.lm_head.weight.detach()) # We do it this way so that no gradient flows through lm_head.weight (basically we freeze lm_head)
            student_log_probs = F.log_softmax(student_logits, dim=-1)

            loss_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none')
            loss_kl = loss_kl.sum(dim=-1) # need to calculate this way because pytorch doesn't really average by (B * T) but instead just (B) or (B * T * C)
            if latent_mask is not None:
                loss_kl = loss_kl[latent_mask].mean()
            else:
                loss_kl = loss_kl.mean()

            loss = loss_next_token + self.lambda_next_h * loss_next_h + self.lambda_kl * loss_kl

        return logits, loss, loss_next_token, loss_next_h, loss_kl

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
