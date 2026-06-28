"""
Evaluation script for Countdown GPT and NextLat models.
Loads a trained checkpoint and evaluates the model's accuracy on the test set.
"""

import os
import sys
import re
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


TEST_FILE = "data/countdown/val_target_b4_t100_n500000.txt" # Evaluation test set
MAX_INTERMEDIATE = 9856
NUM_PAUSE_TOKENS = 8

# Model Architecture (Must match checkpoints config!)
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
BLOCK_SIZE = 32  # Max sequence length for positional embeddings

BATCH_SIZE = 1024  
MAX_NEW_TOKENS = 30
NUM_LOG_SAMPLES = 5  # Number of generations to display in the console
SAVE_RESULTS_PATH = "eval_{model_type}_results.json"



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


def check_countdown_solution(prefix_str, solution_str):
    """
    Verifies that the generated sequence of equations is valid:
    - Consumes starting numbers correctly.
    - Performs correct basic arithmetic.
    - Ends with a result equal to the target.
    """
    parts = prefix_str.strip().split(",")
    if len(parts) < 2:
        return False
    try:
        target = int(parts[-1])
        available_nums = [int(n) for n in parts[:-1]]
    except ValueError:
        return False
        
    equations = [eq.strip() for eq in solution_str.strip().split(",") if eq.strip()]
    if not equations:
        return False
        
    last_val = None
    for eq in equations:
        try:
            left, right_str = eq.split("=")
            right_val = int(right_str)
        except ValueError:
            return False
            
        left_matches = re.match(r"^(\d+)([+\-*/])(\d+)$", left)
        if not left_matches:
            return False
            
        num1 = int(left_matches.group(1))
        op = left_matches.group(2)
        num2 = int(left_matches.group(3))
        
        if num1 not in available_nums:
            return False
        available_nums.remove(num1)
        
        if num2 not in available_nums:
            return False
        available_nums.remove(num2)
        
        if op == "+":
            calc = num1 + num2
        elif op == "-":
            calc = num1 - num2
        elif op == "*":
            calc = num1 * num2
        elif op == "/":
            if num2 == 0 or num1 % num2 != 0:
                return False
            calc = num1 // num2
            
        if calc != right_val:
            return False
            
        available_nums.append(calc)
        last_val = calc
        
    return last_val == target


def generate_batch(model, prefixes, max_new_tokens, eos_token_id, model_type):
    """
    Generates completion tokens autoregressively in a batch using greedy search.
    """
    x = prefixes.clone()
    batch_size = x.shape[0]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
    
    for _ in range(max_new_tokens):
        if finished.all():
            break
            
        with torch.no_grad():
            if model_type == "nextlat":
                logits, _, _, _, _ = model(x)
            else:
                logits, _ = model(x)
                
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Fill finished sequences with EOS
        next_tokens = torch.where(
            finished.unsqueeze(-1),
            torch.tensor(eos_token_id, device=x.device),
            next_tokens
        )
        
        x = torch.cat((x, next_tokens), dim=1)
        finished |= (next_tokens.squeeze(-1) == eos_token_id)
        
    return x


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ["gpt", "nextlat"]:
        print("Usage: python test_countdown.py [gpt|nextlat]")
        sys.exit(1)
        
    model_type = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Model Type: {model_type}")
    
    tokenizer = Tokenizer(max_intermediate=MAX_INTERMEDIATE)
    
    # Load dataset lines
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found at {TEST_FILE}")
        sys.exit(1)
        
    with open(TEST_FILE, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} test lines from {TEST_FILE}")
    
    # Initialize Model Config & Weights
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
            n_embd=N_EMBD
        )
        model = NextLat(config).to(device)
        
    # Load state dict
    checkpoint_path = os.path.join("checkpoints", f"{model_type}_best.pt")
    if not os.path.exists(checkpoint_path):
        # Fallback to latest checkpoint
        checkpoint_path = os.path.join("checkpoints", f"{model_type}_latest.pt")
        
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint path not found. Running evaluation on random weights.")
        
    model.eval()
    
    # Evaluation loop
    correct_count = 0
    total_count = len(lines)
    
    # Collect all prefixes to batch process them
    prefixes_list = []
    prefix_strs = []
    target_sols = []
    
    for line in lines:
        prefix_str, solution_str = line.split("|")
        # Ensure pause tokens are correctly represented at the location of |
        prefix_tokens = tokenizer.encode(prefix_str + "|", NUM_PAUSE_TOKENS)
        
        prefixes_list.append(torch.tensor(prefix_tokens, dtype=torch.long))
        prefix_strs.append(prefix_str)
        target_sols.append(solution_str.replace("<pause>", ""))
        
    # Batch padding for prefixes (since all prefixes have the same length in this task, but we stack them)
    prefixes_tensor = torch.stack(prefixes_list).to(device)
    prefix_len = prefixes_tensor.shape[1]
    
    predictions = []
    results = []
    
    print("Evaluating Model Predictions...")
    for idx in tqdm(range(0, total_count, BATCH_SIZE), desc="Testing"):
        batch_prefixes = prefixes_tensor[idx:idx+BATCH_SIZE]
        
        preds_tensor = generate_batch(
            model=model,
            prefixes=batch_prefixes,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            model_type=model_type
        )
        
        # Keep only the generated portion
        preds_only = preds_tensor[:, prefix_len:].cpu().numpy()
        predictions.extend(preds_only)
        
    # Evaluate correctness
    print("\nEvaluating outputs...")
    logged_samples = 0
    
    for i in range(total_count):
        gen_tokens = predictions[i]
        
        # Decode up to EOS token
        decoded_tokens = []
        for token_id in gen_tokens:
            if token_id == tokenizer.eos_token_id:
                break
            decoded_tokens.append(token_id)
        decoded_clean = tokenizer.decode(decoded_tokens)
        
        correct = check_countdown_solution(prefix_strs[i], decoded_clean)
        if correct:
            correct_count += 1
            
        results.append({
            "prefix": prefix_strs[i],
            "target_sol": target_sols[i],
            "generated": decoded_clean,
            "correct": correct
        })
            
        if logged_samples < NUM_LOG_SAMPLES:
            print("-" * 60)
            print(f"Sample {i+1}:")
            print(f"  Prefix:     {prefix_strs[i]}")
            print(f"  Target Sol: {target_sols[i]}")
            print(f"  Generated:  {decoded_clean}")
            print(f"  Correct:    {correct}")
            logged_samples += 1
            
    accuracy = (correct_count / total_count) * 100
    print("-" * 60)
    print(f"Final Test Accuracy for {model_type.upper()}: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("-" * 60)
    
    # Save detailed outputs
    output_path = SAVE_RESULTS_PATH.format(model_type=model_type)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
