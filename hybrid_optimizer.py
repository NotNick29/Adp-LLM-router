import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

class HybridPromptOptimizer:
    def __init__(self, similarity_threshold: float = 0.75, embedder: SentenceTransformer = None, t5_model=None, t5_tokenizer=None):
        self.similarity_threshold = similarity_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Share T5 and Embedder instances to save VRAM
        if t5_model and t5_tokenizer:
            self.model = t5_model
            self.tokenizer = t5_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small").to(self.device)
            
        self.embedder = embedder if embedder else SentenceTransformer("all-MiniLM-L6-v2")

    def _split_prompt(self, text: str):
        """Splits prompt into background context and core instructions."""
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        if len(sentences) <= 2:
            return text, ""
        context = " ".join(sentences[:-2])
        task = " ".join(sentences[-2:])
        return context, task

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        embs = self.embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def process(self, prompt: str) -> dict:
        words = prompt.strip().split()
        word_count = len(words)

        if word_count <= 150:
            return {
                "action": "FAST_PATH_BYPASS",
                "final_prompt": prompt,
                "reduction": "0.0%",
                "semantic_similarity": 1.0,
                "processed_words": word_count
            }

        # Step 1: Split into Context and Task
        context, task = self._split_prompt(prompt)

        # Step 2: Summarize only the Context
        input_text = "summarize: " + context
        inputs = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=80,
            min_length=25,
            length_penalty=1.5,
            num_beams=2,
            early_stopping=True
        )
        summarized_context = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Step 3: Check context retention similarity
        similarity = self._compute_similarity(context, summarized_context)

        # Step 4: Fuse back together
        if similarity >= self.similarity_threshold:
            fused_prompt = f"{summarized_context}\n\nTask:\n{task}" if task else summarized_context
            out_words = len(fused_prompt.split())
            reduction_pct = ((word_count - out_words) / word_count) * 100
            
            return {
                "action": "HYBRID_ACCEPTED",
                "final_prompt": fused_prompt,
                "reduction": f"{reduction_pct:.1f}%",
                "semantic_similarity": round(similarity, 4),
                "processed_words": out_words
            }
        else:
            return {
                "action": "FALLBACK_TRIGGERED",
                "final_prompt": prompt,
                "reduction": "0.0%",
                "semantic_similarity": round(similarity, 4),
                "processed_words": word_count
            }