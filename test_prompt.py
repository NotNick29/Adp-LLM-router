import re
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from hybrid_optimizer import HybridPromptOptimizer

print("=" * 75)
print("     HYBRID SUMMARIZATION GATEWAY - AUTO-EVALUATOR HARNESS")
print("=" * 75)

# 1. Load shared embedding backbone & Modality Classifier
print("Loading semantic encoder (all-MiniLM-L6-v2) & Modality Classifier...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
modality_model = joblib.load("modality_classifier_dense.joblib")

# 2. Initialize Hybrid Optimizer
print("Initializing Hybrid Optimizer (Split-and-Fuse)...")
hybrid = HybridPromptOptimizer(similarity_threshold=0.75, embedder=embedder)

def evaluate_quality(raw_prompt: str, result: dict) -> dict:
    """
    Automated evaluation engine scoring:
    1. Instruction & Directive Integrity (40 pts)
    2. Compression / Token Efficiency (25 pts)
    3. Semantic Vector Alignment (25 pts)
    4. Fluency & Repetition Check (10 pts)
    """
    final_text = result["final_prompt"]
    action = result["action"]
    sim = result["semantic_similarity"]

    if action == "FAST_PATH_BYPASS":
        return {
            "score": 100.0,
            "verdict": "PERFECT (Fast Path)",
            "metrics": {"Directives": 40, "Efficiency": 25, "Semantics": 25, "Fluency": 10},
            "notes": ["<= 150 words threshold. 100% original text preserved without latency penalty."]
        }

    if action == "FALLBACK_TRIGGERED":
        return {
            "score": 50.0,
            "verdict": "SAFETY FALLBACK (Protected)",
            "metrics": {"Directives": 40, "Efficiency": 0, "Semantics": 0, "Fluency": 10},
            "notes": ["Semantic similarity failed threshold across all retries. Raw prompt safely delivered."]
        }

    # 1. Directive Integrity (Did the task survive verbatim?)
    raw_sentences = re.split(r'(?<=[.?!])\s+', raw_prompt.strip())
    expected_task = " ".join(raw_sentences[-2:]) if len(raw_sentences) > 2 else ""
    directive_score = 40.0 if expected_task in final_text else 20.0

    # 2. Token Efficiency (Ideal target: 35% - 60% reduction)
    raw_words = len(raw_prompt.split())
    out_words = len(final_text.split())
    reduction_pct = ((raw_words - out_words) / raw_words) * 100

    if 35.0 <= reduction_pct <= 65.0:
        eff_score = 25.0
    elif 20.0 <= reduction_pct < 35.0 or 65.0 < reduction_pct <= 75.0:
        eff_score = 18.0
    else:
        eff_score = 10.0

    # 3. Semantic Alignment (Based on cosine similarity vs threshold)
    if sim >= 0.85:
        sem_score = 25.0
    elif sim >= 0.78:
        sem_score = 20.0
    elif sim >= 0.75:
        sem_score = 15.0
    else:
        sem_score = 5.0

    # 4. Fluency & Redundancy Check (Detects repeated phrases)
    words = [w.lower().strip(".,;:()") for w in final_text.split() if len(w) > 4]
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    
    repeated_excess = sum(count - 1 for count in word_counts.values() if count >= 3)
    fluency_score = max(2.0, 10.0 - (repeated_excess * 2.5))

    total_score = round(directive_score + eff_score + sem_score + fluency_score, 1)

    # Diagnostic Notes
    notes = []
    if directive_score == 40.0:
        notes.append("Task directives and constraints preserved 100% intact.")
    else:
        notes.append("Warning: Task instructions partially truncated.")

    notes.append(f"Token reduction: {reduction_pct:.1f}% ({raw_words} -> {out_words} words).")
    notes.append(f"Semantic similarity: {sim:.4f} (Threshold: {hybrid.similarity_threshold}).")

    if repeated_excess > 0:
        notes.append("Notice: Minor phrasing repetition detected in T5 output.")
    else:
        notes.append("Fluency: Clean phrasing without repetitive loops.")

    if total_score >= 85:
        verdict = "EXCELLENT (Production Ready)"
    elif total_score >= 70:
        verdict = "GOOD (Operationally Sound)"
    else:
        verdict = "MARGINAL (Suboptimal Phrasing)"

    return {
        "score": total_score,
        "verdict": verdict,
        "metrics": {
            "Directives": directive_score,
            "Efficiency": eff_score,
            "Semantics": sem_score,
            "Fluency": fluency_score
        },
        "notes": notes
    }

print("\n" + "=" * 75)
print("     LIVE TEST RUNNING WITH AUTOMATED GRADING (Type 'exit' to quit)")
print("=" * 75)

while True:
    prompt = input("\nEnter prompt: ").strip()
    if not prompt:
        continue
    if prompt.lower() in ["exit", "quit", "q"]:
        print("Harness terminated.")
        break

    # STAGE 1: Modality Check
    emb = embedder.encode([prompt], normalize_embeddings=True)
    modality_pred = modality_model.predict(emb)[0]
    modality_prob = modality_model.predict_proba(emb)[0][modality_pred]

    if modality_pred == 1:
        print("\n--- [MODALITY: IMAGE GENERATION] ---")
        print(f"Confidence: {modality_prob * 100:.2f}% | FAST PATH -> Direct to Diffusion Pool.")
        continue

    # STAGE 2: Hybrid Summarization
    res = hybrid.process(prompt, max_retries=3)
    eval_res = evaluate_quality(prompt, res)

    # STAGE 3: Performance & Evaluation Card
    print("\n" + "-" * 75)
    print(f"AUTOMATED AUDIT REPORT  |  Overall Score: {eval_res['score']}/100  [{eval_res['verdict']}]")
    print("-" * 75)
    print(f"  • Directives & Task Safety : {eval_res['metrics']['Directives']}/40 pts")
    print(f"  • Token Reduction Ratio    : {eval_res['metrics']['Efficiency']}/25 pts ({res.get('reduction', '0.0%')})")
    print(f"  • Semantic Alignment       : {eval_res['metrics']['Semantics']}/25 pts (Sim: {res['semantic_similarity']})")
    print(f"  • Linguistic Fluency       : {eval_res['metrics']['Fluency']}/10 pts")
    print(f"  • Optimization Action      : {res['action']} (Retries Used: {res.get('retries_used', 0)})")
    
    print("\n[Diagnostic Notes]:")
    for note in eval_res["notes"]:
        print(f"  -> {note}")

    print("\n[Final Output Forwarded to Model Router]:")
    print(res["final_prompt"])
    print("-" * 75)