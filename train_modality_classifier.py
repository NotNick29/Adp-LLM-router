import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

SAMPLE_SIZE = 752

print("1. Loading raw datasets...")
df_text = pd.read_csv("test_sft.csv", nrows=SAMPLE_SIZE)
text_prompts = df_text["prompt"].dropna().astype(str).tolist()

df_img = pd.read_csv("index.csv", nrows=SAMPLE_SIZE)
img_prompts = df_img["short_prompt"].dropna().astype(str).tolist()

# Add a few standard short factual QA queries to balance length distribution
short_text_anchors = [
    "capital of france", "what is the capital of germany", "who is the president of usa",
    "is a dog an animal or a bird?", "what is 2 + 2?", "explain gravity in simple terms",
    "how to boil an egg", "tell me a joke", "who wrote hamlet?", "what is the weather today?"
]
text_prompts.extend(short_text_anchors)

# Assembly (0: Text, 1: Image)
prompts = text_prompts + img_prompts
labels = [0] * len(text_prompts) + [1] * len(img_prompts)

# 2. Extract Dense Semantic Embeddings using all-MiniLM-L6-v2
print("\n2. Loading 'all-MiniLM-L6-v2' and encoding prompts...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = embedder.encode(prompts, show_progress_bar=True, normalize_embeddings=True)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, labels, test_size=0.20, random_state=42, stratify=labels
)

# 4. Train Classifier
print("\n3. Training Logistic Regression on dense embeddings...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

# 5. Evaluate
y_pred = classifier.predict(X_test)
print("\n" + "=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Text (0)", "Image (1)"]))
print("=" * 50)

# 6. Save Model
print("Saving 'modality_classifier_dense.joblib'...")
joblib.dump(classifier, "modality_classifier_dense.joblib")
print("Done!")