import transformers
from transformers import BertTokenizer
import numpy as np
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

question = "What is the capital of Belgium?"
context = "Belgium is a country in Western Europe. Its capital is Brussels."

inputs = tokenizer(question, context, return_tensors="np", max_length=128, truncation=True, padding="max_length")

np.savetxt("input_ids.txt", inputs["input_ids"], fmt="%d")
np.savetxt("attention_mask.txt", inputs["attention_mask"], fmt="%d")
np.savetxt("token_type_ids.txt", inputs["token_type_ids"], fmt="%d")

print("Bestanden geschreven!")
