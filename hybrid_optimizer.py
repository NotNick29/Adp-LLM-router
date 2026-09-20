import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

class HybridPromptOptimizer:
    def __init__(self, similarity_threshold: float = 0.75, embedder: SentenceTransformer = None):
        self.similarity_threshold = similarity_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing Hybrid T5 Model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
        self.embedder = embedder if embedder else SentenceTransformer("all-MiniLM-L6-v2")

    def _split_prompt(self, text: str):
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        if len(sentences) <= 2:
            return text, ""
        context = " ".join(sentences[:-2])
        task = " ".join(sentences[-2:])
        return context, task

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        embs = self.embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def process(self, prompt: str, max_retries: int = 3) -> dict:
        words = prompt.strip().split()
        word_count = len(words)

        # 1. Fast Path Gating
        if word_count <= 150:
            return {
                "action": "FAST_PATH_BYPASS",
                "final_prompt": prompt,
                "reduction": "0.0%",
                "semantic_similarity": 1.0,
                "retries_used": 0
            }

        # 2. Split Context & Task
        context, task = self._split_prompt(prompt)
        input_text = "summarize: " + context
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        # 3. Validation & Retry Loop
        best_summary = ""
        best_similarity = 0.0
        passed = False
        attempts_taken = 0

        for attempt in range(max_retries):
            attempts_taken += 1
            
            # Dynamic Generation: Try standard beam search first, then increase creativity on retries
            if attempt == 0:
                gen_kwargs = {"num_beams": 2, "early_stopping": True, "length_penalty": 1.5}
            else:
                gen_kwargs = {"do_sample": True, "temperature": 0.7 + (attempt * 0.2), "top_p": 0.9}

            summary_ids = self.model.generate(
                inputs, max_length=80, min_length=25, **gen_kwargs
            )
            summarized_context = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Validation Check
            similarity = self._compute_similarity(context, summarized_context)
            
            if similarity >= self.similarity_threshold:
                best_summary = summarized_context
                best_similarity = similarity
                passed = True
                break  # Exit loop immediately upon success
            else:
                print(f"  [Validator] Attempt {attempts_taken} failed (Sim: {similarity:.4f} < {self.similarity_threshold}). Re-summarizing...")
                if similarity > best_similarity:
                    best_summary = summarized_context
                    best_similarity = similarity

        # 4. Final Fuse and Route
        if passed:
            fused_prompt = f"{best_summary}\n\nTask:\n{task}" if task else best_summary
            out_words = len(fused_prompt.split())
            reduction_pct = ((word_count - out_words) / word_count) * 100
            return {
                "action": "HYBRID_ACCEPTED",
                "final_prompt": fused_prompt,
                "reduction": f"{reduction_pct:.1f}%",
                "semantic_similarity": round(best_similarity, 4),
                "retries_used": attempts_taken
            }
        else:
            return {
                "action": "FALLBACK_TRIGGERED",
                "final_prompt": prompt,
                "reduction": "0.0%",
                "semantic_similarity": round(best_similarity, 4),
                "retries_used": attempts_taken
            }