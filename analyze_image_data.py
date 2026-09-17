import os
import pandas as pd

IMAGE_DATA_FILE = "index.csv"

if not os.path.exists(IMAGE_DATA_FILE):
    raise FileNotFoundError(f"Could not find '{IMAGE_DATA_FILE}'. Make sure it's in the project folder.")

print(f"Loading '{IMAGE_DATA_FILE}'...")
df_img = pd.read_csv(IMAGE_DATA_FILE)

print("\n" + "=" * 50)
print("             IMAGE DATASET PREVIEW")
print("=" * 50)
print(f"Total Rows Found: {len(df_img):,}")
print(f"Columns Available: {list(df_img.columns)}")
print("\nFirst 3 Samples:")
print(df_img.head(3))
print("=" * 50)

# Detect the text column
possible_cols = ["prompt", "caption", "text", "description", "short_prompt"]
target_col = next((col for col in possible_cols if col in df_img.columns), df_img.columns[0])

print(f"\nUsing prompt column: '{target_col}'")
sample_prompts = df_img[target_col].dropna().astype(str)

word_counts = sample_prompts.apply(lambda x: len(x.strip().split()))

print(f"Average Image Prompt Length : {word_counts.mean():.2f} words")
print(f"Max Image Prompt Length     : {word_counts.max()} words")
print(f"Min Image Prompt Length     : {word_counts.min()} words")
print(f"Prompts exceeding 150 words : {(word_counts > 150).sum()} ({(word_counts > 150).mean() * 100:.2f}%)")