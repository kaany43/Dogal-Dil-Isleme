from transformers import AutoTokenizer, AutoModel
import torch

# Model ve tokenizer yükleme
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Örnek metin verisi
metinler = [
    "Doğal dil işleme, bilgisayarların insan dilini anlamasını sağlar.",
    "Makine öğrenimi, verilerden öğrenen algoritmalar geliştirmeyi içerir.",
    "Derin öğrenme, yapay sinir ağlarını kullanarak karmaşık problemleri çözer.",
    "Veri bilimi, büyük veri setlerinden anlamlı bilgiler çıkarmayı amaçlar.",
    "Yapay zeka, insan benzeri zekâya sahip sistemler oluşturmayı hedefler."
]

# Metinleri tokenizasyon
inputs = tokenizer(metinler, return_tensors='pt', padding=True, truncation=True)

#Model ile metin temsilini elde etme
with torch.no_grad():
    outputs = model(**inputs)

# Son katman gizli durumlarını alma
embeddings = outputs.last_hidden_state

# İlk token ([CLS]) temsilini alma
cls_embeddings = embeddings[:, 0, :]
print("Metin Temsilleri (CLS Token):")
for i, metin in enumerate(metinler):
    print(f"Metin: {metin}")
    print(f"Temsil: {cls_embeddings[i].numpy()}\n")
    
# Tüm tokenlerin ortalama temsilini alma
mean_embeddings = torch.mean(embeddings, dim=1)
print("Metin Temsilleri (Ortalama Tokenler):")
for i, metin in enumerate(metinler):
    print(f"Metin: {metin}")
    print(f"Temsil: {mean_embeddings[i].numpy()}\n")

