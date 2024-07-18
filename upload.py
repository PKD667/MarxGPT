import sys,os
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

model = sys.argv[1]

logging.set_verbosity(logging.CRITICAL)

from huggingface_hub import HfFolder

# Save Hugging Face token
HfFolder.save_token(os.environ["HUGGINGFACE_TOKEN"])

model = AutoModelForCausalLM.from_pretrained(model)

model.push_to_hub(sys.argv[2])