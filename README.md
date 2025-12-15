# NLP Çalışmalarım (NLP Studies)

Doğal dil işleme üzerine yaptığım çeşitli çalışmalar ve örnekler.

Natural Language Processing studies and examples I've worked on.

## 📂 Klasörler (Directories)

### `metin-on-isleme/` - Text Preprocessing
Metin temizleme, tokenizasyon, stemming, stop-words işlemleri

- `1-veri-temizleme.py` - Text cleaning (whitespace, punctuation, special chars, HTML)
- `2-tokenizasyon.py` - Tokenization with NLTK
- `3-kok-govde.py` - Stemming and lemmatization
- `4-stop-words.py` - Stop words removal

### `metin-temsili/` - Text Representation
Bag of Words, TF-IDF, Word2Vec, BERT temsilleri

- `1-bag-of-words.py` - Basic Bag of Words implementation
- `2-bow-imdb.py` - BOW with IMDB dataset
- `3-tf-idf.py` - TF-IDF vectorization
- `4-tf-idf-spam.py` - TF-IDF for spam classification
- `5-ngram.py` - N-gram feature extraction
- `6-word-embedding.py` - Word2Vec and FastText embeddings
- `7-word-embedding-imdb.py` - Word embeddings with IMDB data
- `8-transformers-tabanli-metin-temsili.py` - Transformer-based text representations

### `olasiliksal-dil-modelleri/` - Probabilistic Language Models
N-gram modelleri, HMM, Maximum Entropy

- `1-ngram-modelleri.py` - N-gram language models
- `2-hidden-markov-modelleri-1.py` - Hidden Markov Models (basic)
- `3-hidden-markov-modelleri-2.py` - Hidden Markov Models (advanced)
- `4-maximum-entropy-model.py` - Maximum Entropy classification

### `temel-nlp-gorevleri/` - Basic NLP Tasks
Temel NLP görevleri ve uygulamaları

- `1-metin-siniflandirma.py` - Text classification (spam detection)
- `2-named-entity-recognition.py` - Named Entity Recognition with spaCy
- `3-morfolojik-analiz.py` - Morphological analysis
- `4-POS-tagging.py` - Part-of-Speech tagging
- `5-kelime-anlam-belirsizligi-1.py` - Word Sense Disambiguation (WSD) - Lesk algorithm
- `6-kelime-anlam-belirsizligi-2.py` - Word Sense Disambiguation (advanced methods)
- `7-duygu-analizi.py` - Sentiment analysis

### `gelismis-nlp-gorevleri/` - Advanced NLP Tasks
Gelişmiş NLP görevleri

- `1-qa-bert.py` - Question Answering with BERT
- `2-ga-gpt.py` - Text generation with GPT
- `3-ir-bert.py` - Information Retrieval with BERT
- `4-oneri-dl.py` - Deep learning-based recommendation system
- `5-oneri-ml.py` - Machine learning-based recommendation system
- `6-metin-cevirisi.py` - Machine translation with MarianMT
- `7-metin-ozetleme.py` - Text summarization with transformers

### `derin-ogrenme-tabanli-modern-dil-modelleri/` - Deep Learning Language Models
RNN, LSTM, GPT, LLaMA gibi modern dil modelleri

- `1-recurrent-neural-network.py` - RNN for text classification
- `2-long-short-term-memory.py` - LSTM for text generation (Shakespeare)
- `3-gpt-llama.py` - GPT and LLaMA models for text generation

### `data/` - Datasets
Kullanılan veri setleri

- `amazon.csv` - Amazon product reviews
- `IMDB Dataset.csv` - IMDB movie reviews
- `Shakespeare_data.csv` - Shakespeare play dialogues
- `spam.csv` - SMS spam collection

## 🚀 Kurulum (Installation)

```bash
pip install -r requirements.txt
```

## 📝 Notlar (Notes)

- Her dosya bağımsız çalıştırılabilir (Each file can be run independently)
- Veri setleri `data/` klasöründe bulunur (Datasets are in the `data/` folder)
- Bazı modeller internetten indirilir (transformers için) (Some models are downloaded from internet - for transformers)
- GPU varsa daha hızlı çalışır ama CPU'da da çalışır (Faster with GPU, but works on CPU too)
- SpaCy modelleri için: `python -m spacy download en_core_web_sm` (For spaCy models)

## 🔧 Kullandığım Kütüphaneler (Libraries Used)

- **numpy, pandas** - Veri işleme (Data processing)
- **nltk** - Temel NLP işlemleri (Basic NLP operations)
- **scikit-learn** - ML algoritmaları (Machine learning algorithms)
- **gensim** - Word embeddings (Word2Vec, FastText)
- **torch, transformers** - Modern transformer modelleri (Modern transformer models)
- **tensorflow, keras** - Derin öğrenme (Deep learning)
- **spacy** - Gelişmiş NLP (Advanced NLP)
- **matplotlib** - Görselleştirme (Visualization)
- **textblob, beautifulsoup4** - Metin temizleme (Text cleaning)
- **surprise** - Öneri sistemleri (Recommendation systems)
- **pywsd** - Kelime anlam belirsizliği (Word sense disambiguation)

## 📊 Proje Yapısı (Project Structure)

```
dogal-dil-isleme/
├── data/                          # Veri setleri
├── metin-on-isleme/              # Text preprocessing
├── metin-temsili/                # Text representation
├── olasiliksal-dil-modelleri/    # Probabilistic models
├── temel-nlp-gorevleri/          # Basic NLP tasks
├── gelismis-nlp-gorevleri/       # Advanced NLP tasks
├── derin-ogrenme-tabanli-modern-dil-modelleri/  # Deep learning models
├── README.md                     # Bu dosya
├── requirements.txt              # Bağımlılıklar
└── .gitignore                    # Git ignore kuralları
```

Bu benim NLP öğrenme yolculuğumdaki notlarım. 😊

This is my collection of notes from my NLP learning journey. 😊
