import numpy as np
from sentence_transformers import SentenceTransformer
from llmlingua import PromptCompressor


class PromptOptimizer:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

        print("Initializing LLMLingua-2 Token Compressor...")
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
        )

        print("Initializing Semantic Embedding Verifier (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("PromptOptimizer ready!\n")

    def calculate_similarity(self, text_a: str, text_b: str) -> float:
        """Calculates cosine similarity between two texts using normalized dense embeddings."""
        embs = self.embedder.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))

    def process(self, prompt: str, target_rate: float = 0.6) -> dict:
        """
        Executes threshold gating, token compression, and semantic verification.
        target_rate: fraction of tokens to retain (e.g., 0.6 keeps 60%, removes 40%).
        """
        word_count = len(prompt.strip().split())

        # 1. Empirical Threshold Check (<= 150 words bypasses compression)
        if word_count <= 150:
            return {
                "original_prompt": prompt,
                "final_prompt": prompt,
                "action": "FAST_PATH_BYPASS",
                "word_count": word_count,
                "original_tokens": None,
                "compressed_tokens": None,
                "token_reduction": "0.0%",
                "semantic_similarity": 1.0,
                "status": "bypassed_threshold",
            }

        # 2. LLMLingua-2 Token Compression
        result = self.compressor.compress_prompt_llmlingua2(
            prompt,
            rate=target_rate,
            force_reserve_digit=True,
            drop_consecutive=True,
        )

        compressed_text = result["compressed_prompt"]
        orig_tokens = result["origin_tokens"]
        comp_tokens = result["compressed_tokens"]

        # 3. Semantic Verification via Cosine Similarity
        similarity = self.calculate_similarity(prompt, compressed_text)

        # 4. Fallback Gating Logic
        if similarity >= self.similarity_threshold:
            final_prompt = compressed_text
            action_status = "COMPRESSION_ACCEPTED"
        else:
            final_prompt = prompt
            action_status = "FALLBACK_TRIGGERED (Semantic Drift)"

        reduction_pct = (
            (1 - (comp_tokens / orig_tokens)) * 100 if orig_tokens > 0 else 0.0
        )

        return {
            "original_prompt": prompt,
            "final_prompt": final_prompt,
            "action": action_status,
            "word_count": word_count,
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "token_reduction": f"{reduction_pct:.1f}%",
            "semantic_similarity": round(similarity, 4),
            "status": "success",
        }


if __name__ == "__main__":
    optimizer = PromptOptimizer(similarity_threshold=0.85)

    # Test Case 1: Short prompt (<= 150 words -> Fast Path)
    short_input = (
        "Write a Python function to check whether a given string is a palindrome."
    )
    print("=" * 60)
    print("[TEST 1: Short Input]")
    res1 = optimizer.process(short_input)
    print(f"Action Taken        : {res1['action']}")
    print(f"Word Count          : {res1['word_count']}")
    print(f"Semantic Similarity : {res1['semantic_similarity']}")
    print("=" * 60)

    # Test Case 2: Long Context Input (> 150 words -> LLMLingua-2 Pruning)
    long_input = """
    Artificial Intelligence has experienced rapid development over the past decade, 
    evolving from traditional rule-based expert systems into modern deep learning and neural network architectures. 
    One of the major breakthroughs within this domain has been the introduction of the Transformer architecture, 
    which fundamentally changed how natural language processing models operate. Prior to transformers, recurrent neural 
    networks (RNNs) and Long Short-Term Memory (LSTM) networks were the industry standard for sequence modeling tasks. 
    However, RNNs suffered from fundamental limitations, notably catastrophic forgetting and the inability to parallelize 
    computations across sequences due to their sequential hidden state calculations. The Transformer replaced recurrence 
    entirely with multi-head self-attention mechanisms, allowing tokens across an entire document to attend to one another 
    in parallel. This enabled the scaling of models to hundreds of billions of parameters, powering contemporary large 
    language models such as LLaMA, GPT-4, and Claude. In enterprise environments, deploying these foundational models requires 
    rigorous engineering optimizations, including quantization, token distillation, and cost-effective multi-objective routing gateways.
    Explain the mathematical intuition of self-attention and compare it with recurrent neural network layers.
    """

    print("\n" + "=" * 60)
    print("[TEST 2: Long Context Input (> 150 Words)]")
    res2 = optimizer.process(long_input, target_rate=0.6)
    print(f"Action Taken        : {res2['action']}")
    print(f"Word Count          : {res2['word_count']}")
    print(f"Original Tokens     : {res2['original_tokens']}")
    print(f"Compressed Tokens   : {res2['compressed_tokens']}")
    print(f"Token Reduction     : {res2['token_reduction']}")
    print(f"Semantic Similarity : {res2['semantic_similarity']}")
    print("\n[Compressed Text Output]:")
    print(res2["final_prompt"])
    print("=" * 60)