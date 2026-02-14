import torch
import torch.nn as nn
from tqdm import tqdm
import os
from transformers import AutoTokenizer  # Requires: pip install transformers
from model_lora_fine_tune import TinyLlama
from dataloader import AlpacaDataLoader

# 1. Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_SEQ_LEN = 512
LEARNING_RATE = 1e-4
EPOCHS = 1
MODEL_SAVE_PATH = "tiny_llama_alpaca_final.pth"
LOCAL_WEIGHTS = "./tinyllama_local/pytorch_model.bin"
TOKENIZER_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Or local path

# Model Dimensions
VOCAB_SIZE = 32000
DIM = 2048
N_LAYERS = 22
N_HEADS = 32


def train():
    # 2. Initialize Tokenizer & Dataloader
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("Initializing Dataloader...")
    data_factory = AlpacaDataLoader(max_length=MAX_SEQ_LEN)
    train_loader = data_factory.get_dataloader(batch_size=BATCH_SIZE)

    # 3. Initialize Model
    print(f"Building Model on {DEVICE}...")
    model = TinyLlama(vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS).to(DEVICE)

    if os.path.exists(LOCAL_WEIGHTS):
        model.load_pretrained(LOCAL_WEIGHTS)
    else:
        print(f"Warning: Base weights not found. Training from scratch!")

    # 4. Precision & Optimizer Setup
    model = model.to(torch.bfloat16)  # Fast on RTX 50-series

    # Only train LoRA parameters
    trainable_params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=0.01)

    # ignore_index=-100 ensures we don't calculate loss on masked tokens
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Training Loop
    model.train()
    print(f"Starting Training on {len(trainable_params)} LoRA parameters...")

    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            # Ensure labels use -100 for non-target tokens (padding/prompt)
            labels = batch["labels"].to(DEVICE)

            # Mixed Precision Forward Pass
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

                # --- SHIFTING FOR CAUSAL LM ---
                # Logits: [B, Seq-1, Vocab] | Labels: [B, Seq-1]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = criterion(
                    shift_logits.view(-1, VOCAB_SIZE),
                    shift_labels.view(-1)
                )

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            # Stats
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # 6. SHOW INPUT AND OUTPUT (Every 100 steps)
            if batch_idx % 1 == 0:
                with torch.no_grad():
                    # Get predicted IDs
                    preds = torch.argmax(shift_logits, dim=-1)

                    # Pick the first sequence in the batch to show
                    # We use [0, :20] to see the start of the sequence
                    input_text = tokenizer.decode(input_ids[0, :20], skip_special_tokens=False)
                    # We decode labels, but -100 isn't decodable, so we replace it for viewing
                    viewable_labels = shift_labels[0, :20].clone()
                    viewable_labels[viewable_labels == -100] = tokenizer.pad_token_id

                    target_text = tokenizer.decode(viewable_labels, skip_special_tokens=False)
                    pred_text = tokenizer.decode(preds[0, :20], skip_special_tokens=False)

                    print(f"\n\n[STEP {batch_idx}]")
                    print(f"INPUT:  {input_text}")
                    print(f"TARGET: {target_text}")
                    print(f"GUESS:  {pred_text}")
                    print("-" * 50)

            # Intermediate Checkpoint
            if batch_idx % 500 == 0 and batch_idx > 0:
                model.save_fine_tuned_checkpoint(f"lora_checkpoint_{batch_idx}.pth", only_lora=True)

    # 7. Final Save
    print(f"Saving LoRA adapters to {MODEL_SAVE_PATH}")
    model.save_fine_tuned_checkpoint(MODEL_SAVE_PATH, only_lora=True)


if __name__ == "__main__":
    # Blackwell Matmul optimization (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    train()