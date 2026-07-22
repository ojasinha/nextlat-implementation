"""
Minimalistic step-based training script for Countdown GPT and NextLat models.
All hyperparameters are at the top of the file for quick configuration.
"""

import os
import sys
import time
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

TRAIN_FILE = "data/countdown/train_b4_t100_n500000.txt"
VAL_FILE = "data/countdown/val_b4_t100_n500000.txt"
MAX_INTERMEDIATE = 9856 # 64 * 154
NUM_PAUSE_TOKENS = 8

# Model Architecture
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
BLOCK_SIZE = 32         # Max sequence length for positional embeddings

# Optimization & Training
MAX_STEPS = 100000        # Total number of training steps
VAL_INTERVAL = 100       # Evaluate validation loss every VAL_INTERVAL steps
VAL_STEPS = 20           # Number of validation batches to run for evaluation
BATCH_SIZE = 1024
MICRO_BATCH_SIZE = 128
ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
SEED = 108

# NextLat specific weights
LAMBDA_NEXT_H = 2.0
LAMBDA_KL = 1.0
PROJ_FACTOR = 0.5

SAVE_DIR = "checkpoints"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "models")))
from models.GPT import GPT, GPTConfig
from models.NextLat import NextLat, NextLatConfig


class Tokenizer:
    def __init__(self, max_intermediate):
        self.max_intermediate = max_intermediate
        self.encoder = {str(i): i for i in range(max_intermediate)}
        
        # Mapping symbols
        self.encoder["|"] = max_intermediate
        self.encoder["*"] = max_intermediate + 1
        self.encoder["/"] = max_intermediate + 2
        self.encoder["+"] = max_intermediate + 3
        self.encoder["-"] = max_intermediate + 4
        self.encoder["="] = max_intermediate + 5
        self.encoder[","] = max_intermediate + 6
        self.encoder[""] = max_intermediate + 7

        self.decoder = {i: str(i) for i in range(max_intermediate)}
        self.decoder[max_intermediate] = "|"
        self.decoder[max_intermediate + 1] = "*"
        self.decoder[max_intermediate + 2] = "/"
        self.decoder[max_intermediate + 3] = "+"
        self.decoder[max_intermediate + 4] = "-"
        self.decoder[max_intermediate + 5] = "="
        self.decoder[max_intermediate + 6] = ","
        self.decoder[max_intermediate + 7] = ""

        self.numbers = set("0123456789")
        self.eos_token_id = max_intermediate + 7

    @property
    def vocab_size(self):
        return self.max_intermediate + 8

    def encode(self, data, num_pause_tokens=0):
        # Strip out any potential <pause> tokens from the raw text
        data = data.replace("<pause>", "")
        
        out = []
        i = 0
        seen_pipe = False
        while i < len(data):
            if data[i] == "," and not seen_pipe:
                i += 1
                continue
            elif data[i] == "|":
                seen_pipe = True
            
            s = ""
            while i < len(data) and data[i] in self.numbers:
                s += data[i]
                i += 1
            
            if s:
                val = int(s)
                if val >= self.max_intermediate:
                    raise ValueError(f"Value {val} exceeds tokenizer max_intermediate limit {self.max_intermediate}")
                out.append(self.encoder[s])
            elif data[i] == "|":
                for _ in range(num_pause_tokens):
                    out.append(self.encoder["|"])
                i += 1
            else:
                out.append(self.encoder[data[i]])
                i += 1

        return out

    def decode(self, tokens, include_comma=False):
        out = ""
        for token_id in tokens:
            val = token_id.item() if isinstance(token_id, torch.Tensor) else token_id
            if val in self.decoder:
                out += self.decoder[val]
                if include_comma:
                    out += ","
        if include_comma and out.endswith(","):
            out = out[:-1]
        return out

class CountdownDataset(Dataset):
    def __init__(self, file_path, tokenizer, num_pause_tokens):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(file_path, "r") as f:
            self.lines = [line.strip() for line in f if line.strip()]
            
        self.tokenizer = tokenizer
        self.num_pause_tokens = num_pause_tokens
        print(f"Loaded {len(self.lines)} lines from {file_path}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        tokens = self.tokenizer.encode(line, self.num_pause_tokens)
        tokens.append(self.tokenizer.eos_token_id)
        
        L = len(tokens)
        pipe_token_id = self.tokenizer.encoder["|"]
        
        try:
            pipe_start = tokens.index(pipe_token_id)
        except ValueError:
            loss_mask = [False] * (L - 1)
            latent_mask = [False] * (L - 1)
            return tokens, loss_mask, latent_mask
            
        pipe_end = pipe_start + self.num_pause_tokens
        loss_mask = []
        latent_mask = []
        
        for t in range(L - 1):
            is_loss = (pipe_end <= t + 1 < L)
            is_latent = (pipe_start <= t < pipe_end)
            loss_mask.append(is_loss)
            latent_mask.append(is_latent)
            
        return tokens, loss_mask, latent_mask


def collate_fn(batch, pad_token_id):
    max_len = max(len(item[0]) for item in batch)
    xs, ys, loss_masks, latent_masks = [], [], [], []
    for tokens, loss_mask, latent_mask in batch:
        L = len(tokens)
        pad_len = max_len - L
        
        padded_tokens = tokens + [pad_token_id] * pad_len
        x = torch.tensor(padded_tokens[:-1], dtype=torch.long)
        y = torch.tensor(padded_tokens[1:], dtype=torch.long)
        
        padded_loss_mask = loss_mask + [False] * pad_len
        padded_latent_mask = latent_mask + [False] * pad_len
        
        xs.append(x)
        ys.append(y)
        loss_masks.append(torch.tensor(padded_loss_mask, dtype=torch.bool))
        latent_masks.append(torch.tensor(padded_latent_mask, dtype=torch.bool))
        
    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(loss_masks),
        torch.stack(latent_masks)
    )


def infinite_iter(dataloader):
    while True:
        for batch in dataloader:
            yield batch


if len(sys.argv) < 2 or sys.argv[1] not in ["gpt", "nextlat"]:
    print("Usage: python train_countdown.py [gpt|nextlat]")
    sys.exit(1)
    
model_type = sys.argv[1]

# Seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Model Type: {model_type}")

if device == "cuda":
    torch.set_float32_matmul_precision('high')
    
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = torch.bfloat16

# Tokenizer & Dataloaders
tokenizer = Tokenizer(max_intermediate=MAX_INTERMEDIATE)

train_dataset = CountdownDataset(TRAIN_FILE, tokenizer, NUM_PAUSE_TOKENS)
train_loader = DataLoader(
    train_dataset,
    batch_size=MICRO_BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, tokenizer.eos_token_id)
)
train_iter = infinite_iter(train_loader)

val_loader = None
if os.path.exists(VAL_FILE):
    val_dataset = CountdownDataset(VAL_FILE, tokenizer, NUM_PAUSE_TOKENS)
    val_loader = DataLoader(
        val_dataset,
        batch_size=MICRO_BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.eos_token_id)
    )
else:
    print(f"Warning: Validation file not found at {VAL_FILE}. Skipping validation evaluations, but checkpoints will still be saved.")

# Model config & initialization
if model_type == "gpt":
    config = GPTConfig(
        block_size=BLOCK_SIZE,
        vocab_size=tokenizer.vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD
    )
    model = GPT(config).to(device)
else:
    config = NextLatConfig(
        block_size=BLOCK_SIZE,
        vocab_size=tokenizer.vocab_size,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        lambda_next_h=LAMBDA_NEXT_H,
        lambda_kl=LAMBDA_KL,
        proj_factor=PROJ_FACTOR
    )
    model = NextLat(config).to(device)
    
# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=LEARNING_RATE,
    device=device
)

os.makedirs(SAVE_DIR, exist_ok=True)
best_val_loss = float("inf")

# Initialize CSV log file
log_file_path = os.path.join(SAVE_DIR, f"log_{model_type}.csv")
with open(log_file_path, "w") as f:
    if model_type == "gpt":
        f.write("step,loss,dt,val_loss\n")
    else:
        f.write("step,loss,loss_tok,loss_latent,loss_kl,dt,val_loss\n")
        
model.train()

for step in range(1, MAX_STEPS + 1):
    t0 = time.time()
    val_loss_to_log = None
        
    optimizer.zero_grad()
    
    if model_type == "gpt":
        loss_accum = 0.0
        for micro_step in range(ACCUMULATION_STEPS):
            x, y, loss_mask, latent_mask = next(train_iter)
            x, y, loss_mask, latent_mask = (
                x.to(device),
                y.to(device),
                loss_mask.to(device),
                latent_mask.to(device)
            )
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                logits, loss = model(x, y, loss_mask=loss_mask)
                loss_for_backprop = loss / ACCUMULATION_STEPS
            loss_for_backprop.backward()
            loss_accum += loss.detach().item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        t1 = time.time()
        dt = (t1 - t0) * 1000
        
        mean_loss = loss_accum / ACCUMULATION_STEPS
        if step % 10 == 0 or step == 1:
            print(f"step {step:4d} | loss: {mean_loss:.6f} | dt: {dt:.2f}ms")
    else:
        loss_accum = 0.0
        loss_tok_accum = 0.0
        loss_latent_accum = 0.0
        loss_kl_accum = 0.0
        for micro_step in range(ACCUMULATION_STEPS):
            x, y, loss_mask, latent_mask = next(train_iter)
            x, y, loss_mask, latent_mask = (
                x.to(device),
                y.to(device),
                loss_mask.to(device),
                latent_mask.to(device)
            )
            with torch.autocast(device_type=device_type, dtype=ptdtype):
                logits, loss, loss_next_token, loss_next_h, loss_kl = model(
                    x, y, loss_mask=loss_mask, latent_mask=None
                )
                loss_for_backprop = loss / ACCUMULATION_STEPS
            loss_for_backprop.backward()
            loss_accum += loss.detach().item()
            loss_tok_accum += loss_next_token.detach().item()
            loss_latent_accum += loss_next_h.detach().item()
            loss_kl_accum += loss_kl.detach().item()
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        t1 = time.time()
        dt = (t1 - t0) * 1000
        
        mean_loss = loss_accum / ACCUMULATION_STEPS
        mean_loss_tok = loss_tok_accum / ACCUMULATION_STEPS
        mean_loss_latent = loss_latent_accum / ACCUMULATION_STEPS
        mean_loss_kl = loss_kl_accum / ACCUMULATION_STEPS
        if step % 10 == 0 or step == 1:
            print(
                f"step {step:4d} | loss: {mean_loss:.6f} | "
                f"loss_tok: {mean_loss_tok:.6f} | "
                f"loss_latent: {mean_loss_latent:.6f} | "
                f"loss_kl: {mean_loss_kl:.6f} | dt: {dt:.2f}ms"
            )
            
    # Infrequent Validation check
    if step % VAL_INTERVAL == 0:
        # Save latest checkpoint
        checkpoint_path = os.path.join(SAVE_DIR, f"{model_type}_latest.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"--- step {step:4d} | saved checkpoint to {checkpoint_path} ---")
        
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_iter = iter(val_loader)
            with torch.no_grad():
                # Evaluate over similar amount of samples using ACCUMULATION_STEPS
                for _ in range(VAL_STEPS * ACCUMULATION_STEPS):
                    try:
                        vx, vy, vloss_mask, vlatent_mask = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        vx, vy, vloss_mask, vlatent_mask = next(val_iter)
                        
                    vx, vy, vloss_mask, vlatent_mask = (
                        vx.to(device),
                        vy.to(device),
                        vloss_mask.to(device),
                        vlatent_mask.to(device)
                    )
                    
                    if model_type == "gpt":
                        with torch.autocast(device_type=device_type, dtype=ptdtype):
                            _, loss = model(vx, vy, loss_mask=vloss_mask)
                    else:
                        with torch.autocast(device_type=device_type, dtype=ptdtype):
                            _, loss, _, _, _ = model(
                                vx, vy, loss_mask=vloss_mask, latent_mask=None
                            )
                    val_loss += loss.item()
                    val_batches += 1
                    
            avg_val_loss = val_loss / val_batches
            val_loss_to_log = avg_val_loss
            print(f"--- step {step:4d} | validation loss: {avg_val_loss:.6f} ---")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(SAVE_DIR, f"{model_type}_best.pt")
                torch.save(model.state_dict(), best_path)
                print(f"New best validation loss! Saved checkpoints/{model_type}_best.pt")
            model.train()
            
    # Log metrics to CSV
    val_loss_str = f"{val_loss_to_log:.6f}" if val_loss_to_log is not None else ""
    with open(log_file_path, "a") as f:
        if model_type == "gpt":
            f.write(f"{step},{mean_loss:.6f},{dt:.2f},{val_loss_str}\n")
        else:
            f.write(f"{step},{mean_loss:.6f},{mean_loss_tok:.6f},{mean_loss_latent:.6f},{mean_loss_kl:.6f},{dt:.2f},{val_loss_str}\n")