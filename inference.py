import torch
from transformers import AutoTokenizer
from model_with_lora.model_lora import TinyLlama  # Imports your custom LoRA architecture

# 1. Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "tiny_llama_alpaca_final.pth"  # The file saved by train.py
TOKENIZER_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.001


def generate_response(prompt, model, tokenizer, max_new_tokens=100):
    # Format the prompt using the Alpaca template
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(DEVICE)

    # Greedy Decoding Loop
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # Forward pass
            # We use autocast for BF16 speed on your Blackwell card
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(input_ids)

            # Focus only on the last token's predictions
            next_token_logits = logits[:, -1, :]

            # Greedy choice (highest probability)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to the sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Break if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the full sequence and remove the prompt part
    full_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    response = full_text.split("### Response:\n")[-1]
    return response


def main():
    print(f"Loading tokenizer: {TOKENIZER_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # 2. Reconstruct Model Architecture
    print("Initializing model architecture...")
    model = TinyLlama(
        vocab_size=32000,
        dim=2048,
        n_layers=22,
        n_heads=32,
        r=16  # Must match your training rank
    ).to(DEVICE)

    # 3. Load Saved Weights
    print(f"Loading weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    # Check if we saved a dict or just the model
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(torch.bfloat16)  # Optimization for RTX 50-series
    print("Model ready!")

    # 4. Interactive Loop
    while True:
        user_prompt = input("\nUser: ")
        if user_prompt.lower() in ["exit", "quit"]:
            break

        print("Assistant: ", end="", flush=True)
        response = generate_response(user_prompt, model, tokenizer, MAX_NEW_TOKENS)
        print(response)


if __name__ == "__main__":
    main()