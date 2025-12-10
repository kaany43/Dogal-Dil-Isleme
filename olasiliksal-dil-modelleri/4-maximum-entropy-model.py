"""
Classification problemi : Duygu Analizi
"""
#Import necessary libraries
from nltk.classify import MaxentClassifier

#Ornek veri seti (özellikler, etiket)
train_data = [
    ({"love": True, "amazin": True, "happy": True, "terrible": False}, "positive"),
    ({"hate": True, "awful": True, "sad": True, "great": False}, "negative"),
    ({"love": True, "great": True, "fantastic": True, "bad": False}, "positive"),
    ({"hate": True, "terrible": True, "horrible": True, "good": False}, "negative"),
    ({"happy": True, "joyful": True, "pleased": True, "angry": False}, "positive"),
    ({"sad": True, "depressing": True, "unhappy": True, "excited": False}, "negative"),
    ({"love": True, "joyful": True, "fantastic": True, "awful": False}, "positive"),
    ({"hate": True, "sad": True, "horrible": True, "amazin": False}, "negative"),
    ({"happy": True, "great": True, "pleased": True, "terrible": False}, "positive"),
    ({"depressing": True, "angry": True, "unhappy": True, "fantastic": False}, "negative"),
    ({"love": True, "happy": True, "great": True, "horrible": False}, "positive"),
    ({"hate": True, "sad": True, "awful": True, "joyful": False}, "negative")
]

#Maxent Classifier modelini eğitme
maxent_model = MaxentClassifier.train(train_data, max_iter=10)

#Ornek test verileri
test_data = [
    ({"love": True, "fantastic": True, "happy": True, "bad": False}, "positive"),
    ({"hate": True, "terrible": True, "sad": True, "great": False}, "negative"),
    ({"joyful": True, "pleased": True, "amazin": True, "horrible": False}, "positive"),
    ({"depressing": True, "angry": True, "unhappy": True, "fantastic": False}, "negative")
]

#Modeli test etme ve doğruluk hesaplama
correct = 0
for features, label in test_data:
    predicted_label = maxent_model.classify(features)
    print(f"Features: {features} => Predicted: {predicted_label}, Actual: {label}")
    if predicted_label == label:
        correct += 1
accuracy = correct / len(test_data)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

