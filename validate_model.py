import pandas as pd
import joblib

# 1. Load the serialized artifacts
print("Loading trained classifier artifacts...")
vectorizer = joblib.load("modality_vectorizer.joblib")
classifier = joblib.load("modality_classifier.joblib")

# ==========================================
# TEST 1: Unseen Data Validation (5,000 Rows)
# ==========================================
print("\n[TEST 1] Testing on 5,000 completely unseen UltraChat text rows...")
# Skip the first 752 rows that were used for training
df_unseen_text = pd.read_csv("test_sft.csv", skiprows=range(1, 753), nrows=5000)
unseen_prompts = df_unseen_text["prompt"].dropna().astype(str).tolist()

X_unseen = vectorizer.transform(unseen_prompts)
preds = classifier.predict(X_unseen)

# All rows from test_sft are text (Class 0)
correct_text_preds = (preds == 0).sum()
total_unseen = len(preds)
text_accuracy = (correct_text_preds / total_unseen) * 100

print(f"Evaluated Samples : {total_unseen:,}")
print(f"Correctly Tagged as Text (0) : {correct_text_preds:,}")
print(f"Misclassified as Image (1)  : {total_unseen - correct_text_preds:,}")
print(f"Unseen Text Accuracy         : {text_accuracy:.2f}%")

# ==========================================
# TEST 2: Adversarial Edge Cases
# ==========================================
print("\n[TEST 2] Testing adversarial & tricky edge-case prompts...")

edge_cases = [
    # Ambiguous / Tricky Text Prompts (mention art/pictures but want text)
    ("Explain the history of Renaissance art and painting techniques.", "Text (0)"),
    ("Write a Python script using PIL to resize an image to 512x512.", "Text (0)"),
    ("Describe what the Mona Lisa looks like in three paragraphs.", "Text (0)"),
    ("Can you give me prompt ideas for Midjourney to draw a cat?", "Text (0)"),
    
    # Pure Image Prompts
    ("Cyberpunk samurai standing in rain, neon city lights, octane render, 8k --ar 16:9", "Image (1)"),
    ("Macro photography of a dewdrop on a green leaf, bokeh background, soft sunlight", "Image (1)"),
    ("Cute fluffy cartoon monster, 3D Pixar style, vibrant pastel colors, studio lighting", "Image (1)"),
    ("Watercolor illustration of an ancient library with floating books", "Image (1)")
]

print("-" * 75)
print(f"{'Test Prompt':<50} | {'Expected':<10} | {'Predicted':<10} | {'Confidence'}")
print("-" * 75)

for text, expected in edge_cases:
    vec = vectorizer.transform([text])
    pred = classifier.predict(vec)[0]
    prob = classifier.predict_proba(vec)[0][pred]
    
    pred_label = "Image (1)" if pred == 1 else "Text (0)"
    status = "✓" if pred_label == expected else "✗"
    
    truncated_text = (text[:47] + "...") if len(text) > 50 else text
    print(f"{truncated_text:<50} | {expected:<10} | {pred_label:<10} | {prob * 100:5.1f}% {status}")

print("-" * 75)