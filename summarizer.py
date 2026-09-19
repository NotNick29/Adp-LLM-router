import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

class PromptSummarizer:
    def __init__(self, model_name: str = "t5-small", similarity_threshold: float = 0.78, embedder: SentenceTransformer = None):
        self.similarity_threshold = similarity_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading T5 Summarizer ({model_name}) on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Reuse existing embedder to conserve memory
        if embedder is not None:
            self.embedder = embedder
        else:
            print("Loading Semantic Validator (all-MiniLM-L6-v2)...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        embs = self.embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def process(self, prompt: str, max_length: int = 100, min_length: int = 30) -> dict:
        word_count = len(prompt.strip().split())

        # 1. Threshold Gating Check
        if word_count <= 150:
            return {
                "action": "FAST_PATH_BYPASS",
                "final_prompt": prompt,
                "original_words": word_count,
                "processed_words": word_count,
                "reduction": "0.0%",
                "semantic_similarity": 1.0,
                "detail": "<= 150 words threshold. Bypassed to preserve latency."
            }

        # 2. Generative Summarization with T5
        input_text = "summarize: " + prompt.strip()
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary_word_count = len(summary_text.split())

        # 3. Semantic Drift Verification
        similarity = self._compute_similarity(prompt, summary_text)

        # 4. Fallback Verification Gate
        reduction_pct = ((word_count - summary_word_count) / word_count) * 100 if word_count > 0 else 0.0

        if similarity >= self.similarity_threshold and summary_word_count < word_count:
            action = "SUMMARIZATION_ACCEPTED"
            final_output = summary_text
            detail = f"Accepted. Output fluently condensed by {reduction_pct:.1f}%."
        else:
            action = "FALLBACK_TRIGGERED"
            final_output = prompt
            detail = f"Similarity ({similarity:.4f}) < {self.similarity_threshold}. Fallback to raw prompt."

        return {
            "action": action,
            "final_prompt": final_output,
            "original_words": word_count,
            "processed_words": summary_word_count,
            "reduction": f"{reduction_pct:.1f}%",
            "semantic_similarity": round(similarity, 4),
            "detail": detail
        }