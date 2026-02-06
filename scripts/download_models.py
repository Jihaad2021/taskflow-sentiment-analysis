# scripts/download_models.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

print("Downloading models...")

# 1. Sentiment
print("\n1/5 Downloading Sentiment model...")
pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# 2. Emotion
print("\n2/5 Downloading Emotion model...")
pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# 3. Topic (Sentence Transformers)
print("\n3/5 Downloading Topic model...")
SentenceTransformer('all-MiniLM-L6-v2')

# 4. Entity (NER)
print("\n4/5 Downloading Entity model...")
pipeline("ner", model="dslim/bert-base-NER")

# 5. Keyphrase
print("\n5/5 Downloading Keyphrase model...")
pipeline("text2text-generation", model="ml6team/keyphrase-extraction-distilbert-inspec")

print("\n✅ All models downloaded!")