'''
Part of Speech POS : Kelimelerin cümle içindeki görevlerini belirtir. Örneğin, isim, fiil, sıfat gibi kategorilere ayrılırlar. POS etiketleme, metin işleme ve doğal dil işleme uygulamalarında önemli bir adımdır.
'''
#Import necessary libraries
import nltk
from nltk.tag import hmm

#Ornek train veri seti (kelime, POS etiketi)
train_data = [
    [('I', 'PRP'), ('love', 'VBP'), ('apple', 'NN')],
    [('He', 'PRP'), ('likes', 'VBZ'), ('banana', 'NN')],
    [('She', 'PRP'), ('eats', 'VBZ'), ('fruit', 'NN')],
    [('They', 'PRP'), ('love', 'VBP'), ('vegetables', 'NNS')],
    [('We', 'PRP'), ('enjoy', 'VBP'), ('apple', 'NN')],
]

#Train HMM
trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train_supervised(train_data)

#Ornek test veri seti (sadece kelimeler)
test_data = [
    ['I', 'love', 'banana'],
    ['She', 'likes', 'apple'],
    ['They', 'eat', 'fruit'],
    ['We', 'enjoy', 'vegetables'],
]

#POS etiketleme
for sentence in test_data:
    tagged_sentence = hmm_model.tag(sentence)
    print(f"Sentence: {' '.join(sentence)}")
    print(f"Tagged: {tagged_sentence}\n")


