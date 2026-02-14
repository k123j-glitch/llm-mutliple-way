import torch
import torch.nn as nn
from tqdm import tqdm
from model_with_lora.model_lora import TinyLlama  # Your custom model architecture
from dataloader import AlpacaDataLoader  # Your updated dataloader

# 1. Configuration for 16GB VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # You can increase this to 16 given your 16GB VRAM
MAX_SEQ_LEN = 512  # Context window
LEARNING_RATE = 2e-5  # Standard for fine-tuning
EPOCHS = 1
MODEL_SAVE_PATH = "../tiny_llama_alpaca_final.pth"

# Model Dimensions (To match TinyLlama 1.1B style)
VOCAB_SIZE = 32000
DIM = 2048
N_LAYERS = 22
N_HEADS = 32


def train():
    # 2. Initialize Dataloader
    print("Initializing Dataloader...")
    data_factory = AlpacaDataLoader(max_length=MAX_SEQ_LEN)
    train_loader = data_factory.get_dataloader(batch_size=BATCH_SIZE)

    # 3. Initialize Model on GPU
    print(f"Initializing Model on {torch.cuda.get_device_name(0)}...")
    model = TinyLlama(vocab_size=VOCAB_SIZE, dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS).to(DEVICE)

    # Use bfloat16 for RTX 50-series (Faster and more stable)
    model = model.to(torch.bfloat16)

    # 4. Optimizer and Loss Function
    # ignore_index=-100 ensures we don't calculate loss on padding
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 5. Training Loop
    model.train()
    print("Starting Training...")

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward Pass
            # We use autocast for mixed precision (bf16)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

                # --- THE SHIFTING LOGIC ---
                # Shift so that tokens at 'i' predict labels at 'i+1'
                # Logits shape: [Batch, Seq, Vocab]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                # Flatten for CrossEntropy
                loss = criterion(
                    shift_logits.view(-1, VOCAB_SIZE),
                    shift_labels.view(-1)
                )

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Prevents the model from "exploding")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Stats
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Optional: Save checkpoint every 500 batches
            if batch_idx % 500 == 0 and batch_idx > 0:
                torch.save(model.state_dict(), f"checkpoint_{batch_idx}.pth")

    # 6. Final Save
    print(f"Training complete. Saving to {MODEL_SAVE_PATH}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'dim': DIM,
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS
    }, MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()