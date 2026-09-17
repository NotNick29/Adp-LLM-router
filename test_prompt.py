import joblib
from sentence_transformers import SentenceTransformer

print("Loading dense semantic model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
model = joblib.load("modality_classifier_dense.joblib")

print("=" * 60)
print("     SEMANTIC LIVE PROMPT ROUTING (Type 'exit' to quit)")
print("=" * 60)

while True:
    prompt = input("\nEnter a prompt: ").strip()
    if not prompt or prompt.lower() in ["exit", "quit", "q"]:
        break

    # Encode semantically
    embedding = embedder.encode([prompt], normalize_embeddings=True)
    pred = model.predict(embedding)[0]
    prob = model.predict_proba(embedding)[0][pred]

    if pred == 1:
        print(f"-> Detection  : Image Generation (1) [{prob * 100:.1f}%]")
        print(f"-> Action     : Fast Path -> Diffusion Models")
    else:
        word_count = len(prompt.split())
        action = "Compress via LLMLingua-2" if word_count > 150 else "Fast Path (Bypass)"
        print(f"-> Detection  : Text Generation (0) [{prob * 100:.1f}%]")
        print(f"-> Action     : {action} -> Route to LLMs")