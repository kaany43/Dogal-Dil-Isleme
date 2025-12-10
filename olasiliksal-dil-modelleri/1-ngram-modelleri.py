import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from collections import Counter

# Ornek metin verisi
corpus = [
    'I love apple',
    'I love banana',
    'I love fruit apple banana',
    'banana and apple are my favorite fruits',
    'You love apple',
    'He loves banana',
    'They love fruit apple banana',
    'banana and apple are their favorite fruits',
    'We all love fruits',
    'I do not like vegetables',
    'He likes to eat apple',
    'She likes to eat banana',
    'They like to eat fruit apple banana',
    'We do not like vegetables'
]

# Tokenizasyon ve n-gram oluşturma fonksiyonu
def generate_ngrams(corpus, n):
    ngram_list = []
    for sentence in corpus:
        tokens = word_tokenize(sentence.lower())
        ngram_list.extend(ngrams(tokens, n))
    return ngram_list

# N-gram modellerini oluşturma
unigrams = generate_ngrams(corpus, 1)
bigrams = generate_ngrams(corpus, 2)
trigrams = generate_ngrams(corpus, 3)

# N-gram frekanslarını hesaplama
unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)

# Sonuçları yazdırma
print("Unigram Frekansları:")
for gram, freq in unigram_freq.items():
    print(f"{gram}: {freq}")
print("\nBigram Frekansları:")
for gram, freq in bigram_freq.items():
    print(f"{gram}: {freq}")
print("\nTrigram Frekansları:")
for gram, freq in trigram_freq.items():
    print(f"{gram}: {freq}")

# Model Testing
#"I love' bigramindan sonra 'apple' veya 'banana' gelme olasiligi nedir?
test_bigram = ('i', 'love')
prob_apple = trigram_freq[('i', 'love', 'apple')] / bigram_freq[test_bigram] if bigram_freq[test_bigram] > 0 else 0
prob_banana = trigram_freq[('i', 'love', 'banana')] / bigram_freq[test_bigram] if bigram_freq[test_bigram] > 0 else 0
print(f"\nP(apple | 'I love') = {prob_apple}")
print(f"P(banana | 'I love') = {prob_banana}")
