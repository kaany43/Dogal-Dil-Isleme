import nltk

nltk.download('punkt')
nltk.download("punkt_tab")


def tokenize_text(text):
    """
    Metni cümlelere ve kelimelere ayırır.

    Parametreler:
    text (str): Tokenize edilecek metin.

    Dönüş:
    tuple: Cümlelerin ve kelimelerin listesi.
    """
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    return sentences, words

# Örnek kullanım
if __name__ == "__main__":
    sample_text = "Merhaba dünya! Bu bir örnek metindir. Tokenizasyon işlemi yapılacak."
    sentences, words = tokenize_text(sample_text)
    print("Cümleler:", sentences)
    print("Kelimeler:", words)
# Çıktı:
# Cümleler: ['Merhaba dünya!', 'Bu bir örnek metindir.', 'Tokenizasyon işlemi yapılacak.']
# Kelimeler: ['Merhaba', 'dünya', '!', 'Bu', 'bir', 'örnek', 'metindir', '.', 'Tokenizasyon', 'işlemi', 'yapılacak', '.']