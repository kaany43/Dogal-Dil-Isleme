'''
Varlık Adı Tanıma (Named Entity Recognition - NER) Görevi
'''

import spacy
import pandas as pd

# Spacy modelini yükle
nlp = spacy.load("en_core_web_sm")

# Örnek veri seti
data = {
    'text': [
        "Apple is looking at buying U.K. startup for $1 billion",
        "San Francisco considers banning sidewalk delivery robots",
        "London is a big city in the United Kingdom",
        "Elon Musk is the CEO of SpaceX and Tesla"
    ]
}
df = pd.DataFrame(data)

# Varlık adlarını tanıma fonksiyonu
def recognize_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Veri setindeki her metin için varlık adlarını tanıma
df['entities'] = df['text'].apply(recognize_entities)

# Sonuçları yazdır
for index, row in df.iterrows():
    print(f"Text: {row['text']}")
    print("Entities:")
    for entity, label in row['entities']:
        print(f"  - {entity}: {label}")
    print()

