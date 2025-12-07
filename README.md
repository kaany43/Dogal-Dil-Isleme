# Doğal Dil İşleme (NLP) Projesi

Metin ön işleme, metin temsili, olasılıksal dil modelleri ve derin öğrenme tabanlı dil modellerine kapsamlı Python uygulamaları.

## 📁 Proje Yapısı

### `metin-on-isleme/` - Metin Ön İşleme
- **1-veri-temizleme.py**: Boşluk, noktalama, özel karakterler, HTML/URL temizliği
- **2-tokenizasyon.py**: Cümle ve kelime tokenizasyonu
- **3-kok-govde.py**: Stemming (kök bulma) ve Lemmatization (gövde bulma)
- **4-stop-words.py**: Stop-words (durak kelimeleri) çıkarımı

### `metin-temsili/` - Metin Temsili Yöntemleri
- **1-bag-of-words.py**: Bag of Words (BoW) modeli
- **2-bow-imdb.py**: IMDB dataset üzerinde BoW uygulaması
- **3-tf-idf.py**: TF-IDF (Term Frequency-Inverse Document Frequency) modeli
- **4-tf-idf-spam.py**: Spam dataset üzerinde TF-IDF uygulaması
- **5-ngram.py**: N-gram temsili (Unigram, Bigram, Trigram)
- **6-word-embedding.py**: Word2Vec ve FastText kelime gömme yöntemleri
- **7-word-embedding-imdb.py**: IMDB dataset üzerinde Word2Vec uygulaması
- **8-transformers-tabanli-metin-temsili.py**: BERT tabanlı transformer metin temsili

### `olasiliksal-dil-modelleri/` - Olasılıksal Dil Modelleri
- **1-ngram-modelleri.py**: N-gram tabanlı dil modelleri ve olasılık hesaplamaları
- **2-hidden-markov-modelleri-1.py**: HMM ile Part-of-Speech (POS) etiketleme
- **3-hidden-markov-modelleri-2.py**: HMM ile CoNLL2000 veri seti etiketlemesi
- **4-maximum-entropy-model.py**: Maximum Entropy klasifiekatörü ile duygu analizi

### `derin-ogrenme-tabanli-dil-modelleri/` - Derin Öğrenme Modelleri
- **1-recurrent-neural-network.py**: RNN (LSTM/SimpleRNN) ile IMDB sentiment analizi

### `data/` - Veri Setleri
- **IMDB Dataset.csv**: IMDB film yorumları veri seti (50,000 örnek)
- **spam.csv**: Spam/Ham mesaj sınıflandırma veri seti

## 🛠️ Gerekli Kütüphaneler

### Kurulum

```bash
pip install -r requirements.txt
```

### Kullanılan Kütüphaneler:

| Kütüphane | Amaç |
|-----------|------|
| **pandas** | Veri işleme ve analizi |
| **numpy** | Sayısal hesaplamalar |
| **scikit-learn** | Makine öğrenmesi (TF-IDF, CountVectorizer, KMeans) |
| **nltk** | NLP işlemleri (tokenizasyon, POS tagging, HMM) |
| **textblob** | Yazım hatası düzeltme |
| **beautifulsoup4** | HTML/XML ayrıştırma |
| **gensim** | Word2Vec, FastText, LDA modelleri |
| **torch** | PyTorch derin öğrenme framework'ü |
| **transformers** | Hugging Face BERT ve transformer modelleri |
| **tensorflow** | TensorFlow derin öğrenme framework'ü |
| **keras** | Keras API (TensorFlow içine entegre) |
| **matplotlib** | Veri görselleştirme |

## 📚 NLTK Veri İndirmesi

İlk çalıştırmada gerekli NLTK verileri otomatik olarak indirilir:

```python
import nltk
nltk.download('punkt')      # Tokenizasyon
nltk.download('punkt_tab')  # Alternatif tokenizasyon
nltk.download('stopwords')  # Durak kelimeleri
nltk.download('wordnet')    # Lemmatization
nltk.download('averaged_perceptron_tagger')  # POS tagging
nltk.download('conll2000')  # CoNLL2000 veri seti
```

## 🚀 Kullanım

Her Python dosyası bağımsız olarak çalıştırılabilir:

```bash
# Metin Ön İşleme
python metin-on-isleme/1-veri-temizleme.py
python metin-on-isleme/2-tokenizasyon.py
python metin-on-isleme/3-kok-govde.py
python metin-on-isleme/4-stop-words.py

# Metin Temsili
python metin-temsili/1-bag-of-words.py
python metin-temsili/2-bow-imdb.py
python metin-temsili/3-tf-idf.py
python metin-temsili/4-tf-idf-spam.py
python metin-temsili/5-ngram.py
python metin-temsili/6-word-embedding.py
python metin-temsili/7-word-embedding-imdb.py
python metin-temsili/8-transformers-tabanli-metin-temsili.py

# Olasılıksal Dil Modelleri
python olasiliksal-dil-modelleri/1-ngram-modelleri.py
python olasiliksal-dil-modelleri/2-hidden-markov-modelleri-1.py
python olasiliksal-dil-modelleri/3-hidden-markov-modelleri-2.py
python olasiliksal-dil-modelleri/4-maximum-entropy-model.py

# Derin Öğrenme Modelleri
python derin-ogrenme-tabanli-dil-modelleri/1-recurrent-neural-network.py
```

## 📊 Veri Setleri

- **IMDB Dataset**: 50,000 film yorumu (25,000 eğitim, 25,000 test) - Olumlu/Olumsuz sınıflandırması
- **Spam Dataset**: SMS mesajları - Spam/Ham sınıflandırması

## ⚙️ Sistem Gereksinimleri

- **Python**: 3.7+
- **RAM**: Minimum 4GB (özellikle BERT modelleri için)
- **Disk**: Minimum 2GB (veri setleri ve model indirmeleri için)

## 📝 Notlar

- Tüm veri setleri `data/` klasöründe bulunmalıdır
- IMDB ve Spam veri setleri otomatik olarak işlenir
- Transformer modelleri ilk kullanımda indirilir (~500MB)
- RNN eğitimi GPU önerilir ancak CPU'da da çalışır
- UTF-8 kodlaması kullanılmaktadır

## 🔍 Proje Hedefleri

1. **Metin Ön İşleme**: Ham metni işlenebilir formata dönüştürme
2. **Metin Temsili**: Metni sayısal vektörlere dönüştürme
3. **Dil Modelleri**: Metin sınıflandırması ve etiketleme
4. **Derin Öğrenme**: Neural Network tabanlı duygu analizi

## 📄 Lisans

Bu proje eğitim amaçlıdır.
