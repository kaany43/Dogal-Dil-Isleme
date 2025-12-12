'''
Morfolojik analiz, kelimelerin köklerini, eklerini ve diğer morfolojik özelliklerini belirleme sürecidir.
'''
import spacy

# Spacy modelini yükle
nlp = spacy.load("en_core_web_sm")

# Örnek metinler
texts = [
    "The cats are playing with the ball.",
    "She walked quickly to the store.",
    "Unbelievable! They have redesigned the entire system."
]

# Morfolojik analiz fonksiyonu
def morfological_analysis(texts):
    for text in texts:
        print(f"\nMetin: {text}")
        doc = nlp(text)

        for token in doc:
            print(f"Kelime: {token.text}") #Kelimenin kendisi
            print(f"  Kök (lemma): {token.lemma_}") #Kelimenin kök hali
            print(f"  Sözcük türü (POS): {token.pos_}") #Kelimenin dilbilgisel özelliği
            print(f"  Ayrıntılı etiket: {token.tag_}")
            print(f"  Kelime rolü: {token.dep_}")
            print(f"  Morfolojik özellikler: {token.morph}")
            print()

# Fonksiyonu çalıştır
morfological_analysis(texts)