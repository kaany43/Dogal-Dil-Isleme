import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# IMDb veri setini yükleme
data = pd.read_csv('data/IMDB Dataset.csv')

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Sadece harfleri bırakma
    text = text.lower()  # Küçük harfe çevirme
    text = re.sub(r'[^\w\s]', '', text)  # Ozel karakterleri kaldırma
    text = ' '.join([word for word in text.split() if len(word) > 2])  # Kısa kelimeleri kaldırma

    return text
data['cleaned_review'] = data['review'].apply(preprocess_text)

#Tokenizasyon
tokenized_sentences = [simple_preprocess(review) for review in data['cleaned_review']]

#Word2Vec modelini eğitme
word2_vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=5, workers=4, sg=1)

# k-Means kümeleme fonksiyonu
def plot_word_clusters(model, num_clusters=2):
    # Kelime vektörlerini alma
    word_vectors = model.wv
    words = list(word_vectors.index_to_key)[:250]
    vectors = [word_vectors[word] for word in words]

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vectors)
    labels = kmeans.labels_
    # PCA ile 2B'ye indirgeme
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    # Görselleştirme
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(result[:, 0], result[:, 1], c=labels, cmap='viridis')
    plt.title('Word2Vec Kelime Kümeleme Görselleştirmesi')
    plt.xlabel('Componenent 1')
    plt.ylabel('Componenent 2')

    # Kelimeleri etiketleme
    for i, word in enumerate(words):
        plt.text(result[i, 0], result[i, 1], word, fontsize=7)

    plt.colorbar(scatter)
    plt.show()
# Word2Vec modelinin kümeleme görselleştirilmesi
plot_word_clusters(word2_vec_model, num_clusters=2)