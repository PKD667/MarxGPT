import random
import datasets
import os
import sys
import torch


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from huggingface_hub import HfFolder
from trl import SFTTrainer
from peft import LoraConfig

# Save Hugging Face token
HfFolder.save_token(os.environ["HUGGINGFACE_TOKEN"])

# Set verbosity
logging.set_verbosity_error()

# Load dataset
ds = datasets.load_dataset("pkd/marxism", split="train")

# Shuffle dataset
ds = ds.shuffle(seed=random.randint(0, 1000))

# Split into training and validation
ds = ds.train_test_split(test_size=0.1)

# Print dataset structure
print(ds)

# Model and tokenizer details
new_model = "MarxGPT-2-v1"
base_model = "results/checkpoint-16137"
tokenizer_name = "gpt2"

# Define quantization configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Define PEFT configuration
peft_params = LoraConfig(
    target_modules=['q_proj','v_proj',"c_fc","c_attn"],  # Example target modules; adjust based on your model architecture
    task_type="CAUSAL_LM",
)

# Define training arguments
training_params = TrainingArguments(
    num_train_epochs=10,
    per_device_train_batch_size=10,
    gradient_accumulation_steps=2,
    output_dir="./results",
    optim="adamw_torch",
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    max_steps=-1,
    save_strategy="epoch",
    report_to="tensorboard",
)

# Load model
try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.config.use_cache = True
    model.config.pretraining_tp = 1
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    sys.exit(1)

# Initialize trainer
try:
    trainer = SFTTrainer(
        packing=False,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=peft_params,
        dataset_text_field="content",
        tokenizer=tokenizer,
        args=training_params
    )
except Exception as e:
    print(f"Error initializing trainer: {e}")
    sys.exit(1)

# Train model
try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted by user")
    trainer.model.save_pretrained(new_model + "-last")
    sys.exit(1)
except Exception as e:
    trainer.model.save_pretrained(new_model + "-error")
    print(f"Error during training: {e}")
    sys.exit(1)

# Save and push model to hub
try:
    trainer.model.save_pretrained(new_model)
    trainer.model.push_to_hub(new_model)
except Exception as e:
    print(f"Error saving or pushing model: {e}")
    sys.exit(1)
