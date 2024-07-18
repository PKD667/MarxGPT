from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging

model = AutoModelForCausalLM.from_pretrained("MarxGPT-2")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "In order to achieve the liberation of the proletariat, we must"

logging.set_verbosity(logging.CRITICAL)

pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
result = pipe(prompt)
print(result)
print(result[0]['generated_text'])