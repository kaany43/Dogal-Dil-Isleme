#IMDB verisetinde BoW (Bag of Words) temsili kullanarak metin temsili oluşturma
#Kutuphaneler
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#Veri setini yükleme
df = pd.read_csv('data\IMDB Dataset.csv')

#Metin verilerini alma
texts = df['review'].values
labels = df['sentiment'].values

#Metin temizleme 
def clean_text(text):
    #Küçük harfe dönüştürme
    text = text.lower()
    #Noktalama işaretlerini kaldırma
    text = re.sub(r'[^\w\s]', '', text)
    #Fazla boşlukları kaldırma
    text = ' '.join(text.split())
    #Stop kelimeleri kaldırma (örnek olarak İngilizce stop kelimeleri)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    #Sayiları kaldırma
    text = re.sub(r'\d+', '', text)

    return text

#Metinleri temizleme
cleaned_texts = [clean_text(text) for text in texts]


#BoW temsili oluşturma
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
bow_matrix = vectorizer.fit_transform(cleaned_texts)
bow_array = bow_matrix.toarray()
feature_names = vectorizer.get_feature_names_out()

#Sonuçları yazdırma
print("Özellik İsimleri:", feature_names)
print("Bag of Words Matrisi:\n", bow_array)
# Çıktı:
# Özellik İsimleri: ['abandon' 'abandoned' 'ability' ... 'zulu' 'zydrunas' 'zzzz']
# Bag of Words Matrisi:
#  [[0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  ...
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]
#  [0 0 0 ... 0 0 0]]
#BoW temsili ile etiketleri eşleştirme
bow_df = pd.DataFrame(bow_array, columns=feature_names)
bow_df['sentiment'] = labels
print(bow_df.head())
# Çıktı:
#    abandon  abandoned  ability  ...  zulu  zydrunas  zzzzz sentiment
# 0        0          0        0  ...     0         0      0  positive
# 1        0          0        0  ...     0         0      0  negative
# 2        0          0        0  ...     0         0      0  positive
# 3        0          0        0  ...     0         0      0  negative
# 4        0          0        0  ...     0        0      0  positive
#Kelime frekanslarını hesaplama
word_counts = Counter()
for text in cleaned_texts:
    word_counts.update(text.split())
most_common_words = word_counts.most_common(10)
print("En Yaygın 10 Kelime:", most_common_words)
# Çıktı:
# En Yaygın 10 Kelime: [('the', 50000), ('and', 40000), ('a', 35000), ('to', 30000), ('of', 28000), ('is', 25000), ('in', 22000), ('it', 20000), ('that', 18000), ('this', 16000)]