from flask import Flask, request, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
import torch

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained('gpt2', device_map=device, use_cache=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
logging.set_verbosity(logging.CRITICAL)
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=512)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        result = pipe(prompt)
        return render_template('result.html', result=result[0]['generated_text'])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)