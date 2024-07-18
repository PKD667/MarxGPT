import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,  logging
from trl import SFTTrainer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

logging.set_verbosity_error()

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

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=10,
    logging_dir='./logs',
    logging_steps=10,
)

# Function to evaluate a model on validation dataset
def evaluate_checkpoint(model, eval_dataset):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        dataset_text_field="content",
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer.evaluate()

# Function to compute metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = torch.argmax(predictions, dim=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# List all checkpoints
output_dir = "./results"
checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]

best_checkpoint = None
best_metric = float('-inf')

# Evaluate all checkpoints
for checkpoint in checkpoints:
    try:
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
        eval_result = evaluate_checkpoint(model, ds)
        if eval_result["eval_accuracy"] > best_metric:
            best_metric = eval_result["eval_accuracy"]
            best_checkpoint = checkpoint
        print(f"Checkpoint {checkpoint} - Accuracy: {eval_result['eval_accuracy']}")
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint}: {e}")

if best_checkpoint:
    print(f"Best checkpoint: {best_checkpoint} with accuracy: {best_metric}")
else:
    print("No valid checkpoint found.")
