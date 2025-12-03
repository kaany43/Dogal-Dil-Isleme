# Doğal Dil İşleme (NLP) Projesi

Metin ön işleme, metin temsili ve olasılıksal dil modelleri üzerine kapsamlı Python uygulamaları.

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
- **6-word-embedding.py**: Word Embedding (Kelime Gömme) yöntemleri
- **7-word-embedding-imdb.py**: IMDB dataset üzerinde Word Embedding uygulaması
- **8-transformers-tabanli-metin-temsili.py**: Transformer tabanlı metin temsili

### `olasiliksal-dil-modelleri/` - Olasılıksal Dil Modelleri
- **ngram-modelleri.py**: N-gram tabanlı dil modelleri

### `data/` - Veri Setleri
- **IMDB Dataset.csv**: IMDB film yorumları veri seti
- **spam.csv**: Spam sınıflandırma veri seti

## 🛠️ Gerekli Kütüphaneler

```bash
pip install -r requirements.txt
```

### Kullanılan Kütüphaneler:
- **pandas**: Veri işleme ve analizi
- **numpy**: Sayısal hesaplamalar
- **scikit-learn**: Makine öğrenmesi (TF-IDF, CountVectorizer vb.)
- **nltk**: Doğal dil işleme (tokenizasyon, stop-words, stemming, lemmatization)
- **textblob**: Yazım hatası düzeltme
- **beautifulsoup4**: HTML/XML ayrıştırma

## 📚 NLTK Veri İndirmesi

İlk çalıştırmada gerekli NLTK verileri otomatik olarak indirilir:
- `punkt` - Tokenizasyon
- `punkt_tab` - Tokenizasyon alternatifi
- `stopwords` - Durak kelimeleri
- `wordnet` - Lemmatization

## 🚀 Kullanım

Her Python dosyası bağımsız olarak çalıştırılabilir:

```bash
python metin-on-isleme/1-veri-temizleme.py
python metin-temsili/2-bow-imdb.py
python olasiliksal-dil-modelleri/ngram-modelleri.py
```

## 📊 Veri Setleri

- **IMDB Dataset**: Olumlu/olumsuz film yorumları sınıflandırması
- **Spam Dataset**: Spam/Ham mesaj sınıflandırması

## 📝 Notlar

- Tüm veri setleri `data/` klasöründe bulunmalıdır
- Bazı işlemler ilk çalıştırmada biraz zaman alabilir (NLTK indirmeleri)
- UTF-8 kodlaması kullanılmaktadır

## 📄 Lisans

Bu proje eğitim amaçlıdır.