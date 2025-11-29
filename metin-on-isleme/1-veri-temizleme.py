#Metinlerde bulunan fazla boşlukları temizleme
text = "Bu   bir   örnek   metindir.  "
cleaned_text = ' '.join(text.split())
print(cleaned_text)  # Çıktı: "Bu bir örnek metindir."
# %% Buyuk/küçük harf dönüşümü
cased_text = cleaned_text.lower()
print(cased_text)  # Çıktı: "bu bir örnek metindir."

# %% Noktalama işaretlerini temizleme
import string
punctuation_text = "Merhaba, dünya! Bugün nasılsınız?"
cleaned_punctuation_text = punctuation_text.translate(str.maketrans('', '', string.punctuation))
print(cleaned_punctuation_text)  # Çıktı: "Merhaba dünya Bugün nasılsınız"

# %% Ozel karakterleri temizleme
import re
special_char_text = "Merhaba @dunya #2024!"
cleaned_special_char_text = re.sub(r'[^a-zA-Z0-9\s]', '', special_char_text)
print(cleaned_special_char_text)  # Çıktı: "Merhaba dunya 2024"

# %% Yazım hatalarını düzeltme
from textblob import TextBlob
misspelled_text = "Ths is a smple txt with erors."
corrected_text = str(TextBlob(misspelled_text).correct())
print(corrected_text)  # Çıktı: "This is a sample text with errors."

# %% HTML ve URL etiketlerini temizleme
from bs4 import BeautifulSoup
html_text = "<p>Merhaba <a href='http://example.com'>dünya</a>!</p>"
soup = BeautifulSoup(html_text, 'html.parser').get_text()
print(soup)  # Çıktı: "Merhaba dünya!"