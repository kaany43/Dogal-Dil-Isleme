import nltk
from nltk.tag import hmm
from nltk.corpus import conll2000

# Veri setini indir (ilk kez çalıştırıldığında)
nltk.download('conll2000')

# Conll2000 veri setinden eğitim ve test verilerini yükleme
train_data = conll2000.tagged_sents('train.txt')
test_data = conll2000.tagged_sents('test.txt')

# HMM modelini eğitme
trainer = hmm.HiddenMarkovModelTrainer()
hmm_model = trainer.train(train_data)

# Yeni test cumlesi ve test
test_sentences = [
    ['The', 'dog', 'barked'],
    ['A', 'cat', 'meowed'],
    ['The', 'sun', 'is', 'shining'],
    ['They', 'are', 'playing', 'football']
]

for sentence in test_sentences:
    tagged_sentence = hmm_model.tag(sentence)
    print(f"Sentence: {' '.join(sentence)}")
    print(f"Tagged: {tagged_sentence}\n")

# Modelin doğruluğunu değerlendirme
accuracy = hmm_model.accuracy(test_data)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
