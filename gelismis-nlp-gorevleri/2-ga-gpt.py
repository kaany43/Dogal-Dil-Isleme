from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import warnings
warnings.filterwarnings("ignore")

# ------------------ DEVICE ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------ MODEL ------------------
model_name = "gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# GPT-2'de pad token yok → eos'u pad olarak ayarla
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

# ------------------ GENERATE FUNCTION ------------------
def generate_answer(context, question):

    prompt = (
        "Answer the question using ONLY the given context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=80,          # 🔥 kontrol
            temperature=0.6,            # daha az saçmalama
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,     # 🔥 tekrar engelle
            repetition_penalty=1.3,     # 🔥 loop kırıcı
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Sadece Answer kısmını al
    if "Answer:" in decoded:
        decoded = decoded.split("Answer:")[-1]

    return decoded.strip()


# ------------------ TEST ------------------
question = "What is the capital of France?"
context = """
-Paris (French) Capital - 1st Century AD/13th century A to Z The following are some examples from Europe and North America in which we can find capitals that date back as far after 1350AD or more; they may be found on this site but have been removed because their meaning was unclear by others such data would not fit with our understanding... These were founded around 1560
"""

print("Answer:", generate_answer(context, question))


question = "What is Machine Learning?"
context = """
, The term "machine" refers specifically or explicitly in this article's Introduction that it means various things including artificial intelligence; machine vision/visioning technologies for example as well advanced mathematical modeling techniques such up-to four years ago ; Artificial Intelligence applications where computers can be used by people who have never met one other person but are able learn from them through experience , etc. : AI technology based upon
"""

print("Answer:", generate_answer(context, question))
