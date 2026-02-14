import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class AlpacaDataLoader:
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Llama models don't have a default pad token, so we use the EOS token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def format_alpaca(self, example):
        """Converts instruction/input/output into a single string."""
        if example.get("input"):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return text

    def get_dataloader(self, batch_size=4, split="train[:1000]"):
        dataset = load_dataset("tatsu-lab/alpaca", split=split)

        def tokenize_function(examples):
            # 1. Format text
            formatted_texts = [self.format_alpaca(dict(zip(examples.keys(), values)))
                               for values in zip(*examples.values())]

            # 2. Tokenize
            outputs = self.tokenizer(
                formatted_texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )

            # 3. Add Labels
            # In Causal LM, labels are exactly the input_ids.
            # We use -100 for padding tokens so the loss function ignores them.
            labels = outputs["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            outputs["labels"] = labels

            return outputs

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        tokenized_dataset.set_format("torch")

        return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)


# Test the loader
if __name__ == "__main__":
    loader_factory = AlpacaDataLoader()
    loader = loader_factory.get_dataloader(batch_size=2)
    batch = next(iter(loader))

    print(f"Batch keys: {batch.keys()}")  # Now includes 'labels'
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")

    # Check if padding is masked in labels (should see -100s)
    print(f"First 10 labels of row 0: {batch['labels'][0][:10]}")