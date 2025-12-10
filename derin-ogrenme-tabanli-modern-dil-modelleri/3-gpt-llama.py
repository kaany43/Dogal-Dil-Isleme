'''
Metin üretimi için GPT ve LLaMA modellerini karşılaştıran bir Python betiği.
Bu betik, Hugging Face'in Transformers kütüphanesini kullanarak her iki modeli de yükler ve aynı başlangıç metniyle metin üretir.
'''

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Cihazı belirle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPT-2 Modeli ve Tokenizer'ı yükle
gpt_model_name = "gpt2"
gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_name)
gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_name).to(device)
gpt_model.eval()

# Use a pipeline as a high-level helper
from transformers import pipeline
llama_model_name = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline("text-generation", model=llama_model_name, device=0 if torch.cuda.is_available() else -1)

# LLaMA Modeli ve Tokenizer'ı yükle

llama_tokienizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name).to(device)
llama_model.eval()

# Başlangıç metni
prompt = "Once upon a time in a land far, far away"

### GPT-2 ile metin üretimi
#Tokenizasyon
gpt_inputs = gpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
# Metin üretimi
gpt_outputs = gpt_model.generate(gpt_inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
gpt_generated_text = gpt_tokenizer.decode(gpt_outputs[0], skip_special_tokens=True)

### LLaMA ile metin üretimi
llama_outputs = pipe(prompt, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
llama_generated_text = llama_outputs[0]['generated_text']

# Sonuçları yazdır
print("GPT-2 Generated Text:\n", gpt_generated_text)
print("\nLLaMA Generated Text:\n", llama_generated_text)

