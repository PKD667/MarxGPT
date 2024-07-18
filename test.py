from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import sys
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(sys.argv[1],device_map=device,use_cache=True)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = input("Enter a prompt: ")

logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)
result = pipe(prompt)
print("---")
print(result[0]['generated_text'])
print("---")
