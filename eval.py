import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,  logging, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from sklearn.metrics import accuracy_score

logging.set_verbosity_error()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load dataset
ds = load_dataset("pkd/marxism", split="train")
ds = ds.shuffle(seed=1917)
# keep only 1000 examples for faster training
ds = ds.select(range(1000))

# Define model and tokenizer
tokenizer_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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
training_args = TrainingArguments(
    num_train_epochs=5,
    per_device_eval_batch_size=10,
    gradient_accumulation_steps=1,
    output_dir="./results",
    optim="adamw_torch",
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    max_steps=-1
)

# Function to evaluate a model on validation dataset
def evaluate_checkpoint(model, eval_dataset):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        dataset_text_field="content",
        tokenizer=tokenizer,
        peft_config=peft_params,
    )
    return trainer.evaluate()

# List all checkpoints
output_dir = "./results"
checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]

best_checkpoint = None
best_metric = float('inf')

# Evaluate all checkpoints
for checkpoint in checkpoints:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=quant_config,
            device_map="auto"
        )
        eval_result = evaluate_checkpoint(model, ds)
        if eval_result["eval_loss"] < best_metric:
            best_metric = eval_result["eval_loss"]
            best_checkpoint = checkpoint
        print(f"Checkpoint {checkpoint} - Loss: {eval_result['eval_loss']}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint}: {e}")

if best_checkpoint:
    print(f"Best checkpoint: {best_checkpoint} with loss: {best_metric}")
else:
    print("No valid checkpoint found.")
