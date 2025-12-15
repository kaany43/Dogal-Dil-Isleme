from transformers import BertTokenizer, BertForQuestionAnswering
import torch
import warnings
warnings.filterwarnings("ignore")

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# squad veri seti uzerinde ince ayar yapilmis bert modeli
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# model
model = BertForQuestionAnswering.from_pretrained(model_name)
model.to(device)
model.eval()

def predict_answer(context, question):
    """
    context = metin
    question = soru
    """

    encoding = tokenizer.encode_plus(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    #TENSORLERI GPU'YA AL
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        start_scores, end_scores = model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

    start_index = torch.argmax(start_scores, dim=1).item()
    end_index = torch.argmax(end_scores, dim=1).item()

    answer_tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0][start_index:end_index + 1]
    )

    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer


# ------------------------------------

question = "What is the capital of France"
context = "paris"

print("Answer:", predict_answer(context, question))


question = "What is Machine Learning?"
context = """
the scientific study of algorithms and statistical models
"""

print("Answer:", predict_answer(context, question))
