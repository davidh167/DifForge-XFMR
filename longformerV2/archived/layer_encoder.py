"""
Deprecated in favor of layerencoder2.py
"""

from datasets import load_dataset
from transformers import LongformerTokenizer, LongformerForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os

# Load your preprocessed layer dataset (each line is one 1024-token layer)
dataset = load_dataset("text", data_files={"train": "layer_dataset.txt"})

# Optionally, split into train/evaluation
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Load a Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
# Add custom special tokens if needed
special_tokens = {
    "pad_token": "[PAD]",
    "bos_token": "[START_LAYER]",
    "eos_token": "[END_LAYER]",
    "additional_special_tokens": ["[END_ROW]"]
}
tokenizer.add_special_tokens(special_tokens)
print("Tokenizer vocab size:", len(tokenizer))

def tokenize_function(examples):
    # Each example is a layer string; we truncate/pad to 1024 tokens.
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval  = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)

training_args = TrainingArguments(
    output_dir="./longformer_layer_encoder",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,    # Adjust depending on your GPU
    per_device_eval_batch_size=2,
    num_train_epochs=3,               # For 2000 structures × 32 layers, you'll have ~64k training examples
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,                        # Mixed precision to speed up training and reduce memory usage
)

model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096")
model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings for new tokens

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained("./longformer_layer_encoder")
tokenizer.save_pretrained("./longformer_layer_encoder")
