import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from gensim.models import Word2Vec, fasttext
from gensim.utils import simple_preprocess

# Ornek metin verisi
metinler = [
    "Doğal dil işleme, bilgisayarların insan dilini anlamasını sağlar.",
    "Makine öğrenimi, verilerden öğrenen algoritmalar geliştirmeyi içerir.",
    "Derin öğrenme, yapay sinir ağlarını kullanarak karmaşık problemleri çözer.",
    "Veri bilimi, büyük veri setlerinden anlamlı bilgiler çıkarmayı amaçlar.",
    "Yapay zeka, insan benzeri zekâya sahip sistemler oluşturmayı hedefler."
]

tokenized_sentences = [simple_preprocess(metin) for metin in metinler]

#Word2Vec modelini eğitme
word2_vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# FastText modelini eğitme
fasttext_model = fasttext.FastText(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Kelime vektörlerini alma, PCA ve görselleştirme fonksiyonu
def plot_word_embedding(model, title):
    word_vectors = model.wv
    words = list(word_vectors.index_to_key)[:1000]
    vectors = [word_vectors[word] for word in words]

    pca = PCA(n_components=3)
    result = pca.fit_transform(vectors)

    fig =  plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(result[:, 0], result[:, 1], result[:, 2])
    for i, word in enumerate(words):
        ax.text(result[i, 0], result[i, 1], result[i, 2], word)
    ax.set_title(title)
    ax.set_xlabel('Componenent 1')
    ax.set_ylabel('Componenent 2')
    ax.set_zlabel('Componenent 3')
    plt.show()

# Word2Vec modelinin görselleştirilmesi
plot_word_embedding(word2_vec_model, "Word2Vec Kelime Gömme Görselleştirmesi")
# FastText modelinin görselleştirilmesi
plot_word_embedding(fasttext_model, "FastText Kelime Gömme Görselleştirmesi")
    