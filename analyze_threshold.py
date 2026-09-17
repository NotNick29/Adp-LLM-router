import os
import pandas as pd
import numpy as np

# 1. Path Configuration
DATA_FILE = "test_sft.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}' in the project directory. "
        "Ensure the CSV file is placed in the root folder."
    )

print(f"Loading dataset from '{DATA_FILE}'...")
df = pd.read_csv(DATA_FILE)

# 2. Identify the Text Column
# UltraChat typically labels this column 'prompt', fallback to column 0 if mismatched
target_col = "prompt" if "prompt" in df.columns else df.columns[0]
print(f"Using column '{target_col}' for length distribution analysis.")

# 3. Compute Length Metrics (Word Count & Character Count)
print("Analyzing token and word count distributions across all rows...")
cleaned_series = df[target_col].dropna().astype(str)

# Calculate word counts (whitespace tokenization)
word_counts = cleaned_series.apply(lambda text: len(text.strip().split()))

# Calculate summary statistics
total_samples = len(word_counts)
mean_wc = word_counts.mean()
std_wc = word_counts.std()
min_wc = word_counts.min()
max_wc = word_counts.max()

# Percentiles: 25th, 50th (median), 75th, 90th, 95th, 99th
p25, p50, p75, p90, p95, p99 = np.percentile(word_counts, [25, 50, 75, 90, 95, 99])

# 4. Print Distribution Report
print("\n" + "=" * 55)
print("       EMPIRICAL PROMPT LENGTH DISTRIBUTION")
print("=" * 55)
print(f"Total Evaluated Samples : {total_samples:,}")
print(f"Minimum Word Count      : {min_wc} words")
print(f"Maximum Word Count      : {max_wc:,} words")
print(f"Mean Word Count         : {mean_wc:.2f} words (Std Dev: {std_wc:.2f})")
print("-" * 55)
print(f"25th Percentile (Q1)    : {p25:.1f} words")
print(f"50th Percentile (Median): {p50:.1f} words")
print(f"75th Percentile (Q3)    : {p75:.1f} words")
print(f"90th Percentile         : {p90:.1f} words")
print(f"95th Percentile         : {p95:.1f} words")
print(f"99th Percentile         : {p99:.1f} words")
print("=" * 55)

# 5. Evaluate Candidate Threshold Cut-offs
candidate_thresholds = [100, 120, 150, 180, 200]

print("\n" + "=" * 55)
print("     THRESHOLD GATING TRADE-OFF ANALYSIS")
print("=" * 55)
print(f"{'Threshold':<12} | {'Bypassed (Fast Path)':<22} | {'Compressed (LLMLingua-2)':<20}")
print("-" * 55)

for th in candidate_thresholds:
    compressed_count = (word_counts > th).sum()
    bypassed_count = total_samples - compressed_count
    
    pct_compressed = (compressed_count / total_samples) * 100.0
    pct_bypassed = (bypassed_count / total_samples) * 100.0
    
    print(f"> {th:<10} words | {pct_bypassed:>6.2f}% ({bypassed_count:>6,} rows) | {pct_compressed:>6.2f}% ({compressed_count:>6,} rows)")

print("=" * 55)