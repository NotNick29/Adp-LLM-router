import joblib
from sentence_transformers import SentenceTransformer
from compressor import PromptOptimizer
from summarizer import PromptSummarizer
from hybrid_optimizer import HybridPromptOptimizer

print("=" * 85)
print("     INITIALIZING ROUTER GATEWAY: 3-WAY PREPROCESSING COMPARISON")
print("=" * 85)

# 1. Load shared embedding backbone and Modality Classifier
print("Loading semantic encoder (all-MiniLM-L6-v2) & Modality Classifier...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
modality_model = joblib.load("modality_classifier_dense.joblib")

# 2. Initialize subsystems (sharing memory models)
print("1/3 Loading Token Compressor (LLMLingua-2)...")
compressor = PromptOptimizer(similarity_threshold=0.85)

print("2/3 Loading Pure Summarizer (T5-small)...")
summarizer = PromptSummarizer(similarity_threshold=0.78, embedder=embedder)

print("3/3 Loading Hybrid Optimizer (Split-and-Fuse)...")
hybrid = HybridPromptOptimizer(
    similarity_threshold=0.75,
    embedder=embedder,
    t5_model=summarizer.model,
    t5_tokenizer=summarizer.tokenizer
)

print("\n" + "=" * 85)
print("     LIVE 3-WAY COMPARISON TEST (Type 'exit' to quit)")
print("=" * 85)

while True:
    prompt = input("\nEnter prompt: ").strip()
    if not prompt:
        continue
    if prompt.lower() in ["exit", "quit", "q"]:
        print("Harness terminated.")
        break

    # -------------------------------------------------------------
    # STAGE 1: Modality Classification Gate
    # -------------------------------------------------------------
    emb = embedder.encode([prompt], normalize_embeddings=True)
    modality_pred = modality_model.predict(emb)[0]
    modality_prob = modality_model.predict_proba(emb)[0][modality_pred]

    print("\n--- [1. MODALITY DETECTION] ---")
    if modality_pred == 1:
        print(f"Modality Detection  : Image Generation (Class 1)")
        print(f"Confidence Score    : {modality_prob * 100:.2f}%")
        print(f"Routing Strategy    : FAST PATH -> Direct to Diffusion Pool")
        print(f"Action Detail       : Preprocessing bypassed to preserve visual render tags.")
        continue

    print(f"Modality Detection  : Text Generation (Class 0)")
    print(f"Confidence Score    : {modality_prob * 100:.2f}%")

    # -------------------------------------------------------------
    # STAGE 2: Execute All 3 Pipelines in Parallel
    # -------------------------------------------------------------
    comp_res = compressor.process(prompt, target_rate=0.6)
    summ_res = summarizer.process(prompt)
    hybr_res = hybrid.process(prompt)

    word_count = len(prompt.split())
    print(f"\n--- [2. PREPROCESSING 3-WAY COMPARISON (Input: {word_count} words)] ---")

    # Clean comparative table
    row_fmt = "{:<20} | {:<20} | {:<20} | {:<20}"
    print(row_fmt.format("Metric", "Compressor (Lingua)", "Pure T5", "Hybrid (Split-Fuse)"))
    print("-" * 88)
    print(row_fmt.format("Action Status", comp_res['action'], summ_res['action'], hybr_res['action']))
    print(row_fmt.format("Reduction Rate", comp_res.get('token_reduction', '0.0%'), summ_res['reduction'], hybr_res['reduction']))
    print(row_fmt.format("Semantic Sim", str(comp_res['semantic_similarity']), str(summ_res['semantic_similarity']), str(hybr_res['semantic_similarity'])))
    print(row_fmt.format("Output Words", str(comp_res.get('compressed_tokens', word_count)), str(summ_res['processed_words']), str(hybr_res['processed_words'])))

    # -------------------------------------------------------------
    # STAGE 3: Final Delivered Text Outputs
    # -------------------------------------------------------------
    print("\n--- [3. OUTPUT TEXTS DELIVERED TO DOWNSTREAM LLM] ---")
    
    print("\n>>> [OPTION 1: COMPRESSOR (LLMLingua-2)]:")
    print(comp_res["final_prompt"] if comp_res["action"] != "FALLBACK_TRIGGERED" else "[FALLBACK TO RAW PROMPT]")

    print("\n>>> [OPTION 2: PURE SUMMARIZER (T5-small)]:")
    print(summ_res["final_prompt"] if summ_res["action"] != "FALLBACK_TRIGGERED" else "[FALLBACK TO RAW PROMPT]")

    print("\n>>> [OPTION 3: HYBRID (Split-and-Fuse)]:")
    print(hybr_res["final_prompt"] if hybr_res["action"] != "FALLBACK_TRIGGERED" else "[FALLBACK TO RAW PROMPT]")