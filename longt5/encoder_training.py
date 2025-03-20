from datasets import load_dataset
from transformers import T5Tokenizer, LongT5Model, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------------------
# 1️⃣ Load Dataset
# ------------------------------
dataset = load_dataset("text", data_files={"train": "layer_dataset.txt"})

# Split into train/evaluation (90% train, 10% eval)
split_dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# ------------------------------
# 2️⃣ Load and Customize Tokenizer
# ------------------------------
tokenizer = T5Tokenizer.from_pretrained("google/long-t5-local-base")

# Define custom special tokens
special_tokens = {
    "pad_token": "[PAD]",
    "bos_token": "[START_LAYER]",
    "eos_token": "[END_LAYER]",
    "additional_special_tokens": ["[END_ROW]"]
}

tokenizer.add_special_tokens(special_tokens)
print("Tokenizer vocab size:", len(tokenizer))

# Token ID for "air" token (used for ignoring loss)
air_token_id = tokenizer.convert_tokens_to_ids("air")
print("Air token id:", air_token_id)

# ------------------------------
# 3️⃣ Tokenization Function for Seq2Seq
# ------------------------------
def tokenize_function(examples):
    # Tokenize all texts at once
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    
    # Prepare new lists to hold our processed fields
    new_input_ids = []
    new_decoder_input_ids = []
    new_labels = []
    new_attention_mask = []
    
    # Process each example in the batch
    for input_ids, att_mask in zip(model_inputs["input_ids"], model_inputs["attention_mask"]):
        # Ensure input_ids is exactly length 1024 (tokenizer should have already padded/truncated)
        input_ids = input_ids[:1024]
        
        # Create labels by shifting right: prepend pad_token_id and remove last token.
        labels = [tokenizer.pad_token_id] + input_ids[:-1]
        
        # Append processed values
        new_input_ids.append(input_ids)
        new_decoder_input_ids.append(labels)
        new_labels.append(labels)
        new_attention_mask.append(att_mask[:1024])
    
    # Replace the original fields with the processed ones
    model_inputs["input_ids"] = new_input_ids
    model_inputs["decoder_input_ids"] = new_decoder_input_ids
    model_inputs["labels"] = new_labels
    model_inputs["attention_mask"] = new_attention_mask
    
    return model_inputs


sample = train_dataset[0]  # Get the first example
print("DEBUG - Raw dataset sample:", sample)

tokenized_sample = tokenize_function(sample)
print("DEBUG - Tokenized sample:", tokenized_sample)

# Tokenize the dataset
tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ------------------------------
# 4️⃣ Data Collator for Seq2Seq
# ------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="google/long-t5-local-base")

# ------------------------------
# 5️⃣ Training Configuration
# ------------------------------
training_args = TrainingArguments(
    output_dir="./longt5_layer_encoder",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,    # Adjust depending on GPU
    per_device_eval_batch_size=1,
    num_train_epochs=3,               # Adjust epochs based on dataset size
    weight_decay=0.01,
    save_strategy="epoch",
    fp16=True,                        # Mixed precision training
)

# ------------------------------
# 6️⃣ Load LongT5 Model
# ------------------------------
model = LongT5Model.from_pretrained("google/long-t5-local-base")
model.gradient_checkpointing_enable()
model.resize_token_embeddings(len(tokenizer))  # Adjust embeddings for new tokens

# ------------------------------
# 7️⃣ Custom Trainer to Ignore "Air" Tokens
# ------------------------------
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        if labels is None:
            print("WARNING: Labels are missing from inputs!")
            return torch.tensor(0.0, device=model.device)  # Return zero loss to prevent crash

        labels = torch.tensor(labels, device=model.device)  # Ensure it's a tensor
        labels[labels == air_token_id] = -100  # Ignore "air" tokens

        with torch.amp.autocast("cuda"):
            outputs = model(**inputs)
            logits = outputs.last_hidden_state

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# ------------------------------
# 8️⃣ Train Model
# ------------------------------
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# ------------------------------
# 9️⃣ Save Trained Model & Tokenizer
# ------------------------------
model.save_pretrained("./longt5_layer_encoder")
tokenizer.save_pretrained("./longt5_layer_encoder")