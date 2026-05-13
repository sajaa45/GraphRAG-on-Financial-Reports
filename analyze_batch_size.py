"""
Analyze the actual token/character count when batching chunks for LLM extraction.
This helps determine if batching all chunks is safe or if we risk truncation.
"""

import os
import sys
import json
from typing import List, Dict
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'KG_building'))

load_dotenv()

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English"""
    return len(text) // 4

def analyze_batch_size(collection_name: str = "financial_docs",
                       keywords: List[str] = None,
                       n_chunks: int = 10,
                       threshold: float = 0.15):
    
    if keywords is None:
        keywords = ['net income', 'revenue']
    
    print(f"{'='*80}")
    print(f"BATCH SIZE ANALYSIS")
    print(f"{'='*80}")
    print(f"Collection: {collection_name}")
    print(f"Keywords: {keywords}")
    print(f"Chunks per keyword: {n_chunks}")
    print(f"Similarity threshold: {threshold}")
    print()
    
    # Initialize
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = QdrantClient(host="localhost", port=6333)
    
    all_batches = []
    
    for kw_idx, kw in enumerate(keywords, 1):
        print(f"\n{'='*80}")
        print(f"KEYWORD {kw_idx}/{len(keywords)}: '{kw}'")
        print(f"{'='*80}")
        
        # Get chunks
        emb = embedding_model.encode([kw])[0].tolist()
        results = client.query_points(
            collection_name=collection_name,
            query=emb,
            query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="chunk"))]),
            limit=n_chunks,
            with_payload=True,
        ).points
        
        chunks = [
            {
                'text': r.payload['text'],
                'similarity': r.score,
                'section': r.payload.get('section_title', 'Unknown'),
                'page': r.payload.get('source_page', 'N/A')
            }
            for r in results if r.score >= threshold
        ]
        
        if not chunks:
            print(f"  ✗ No chunks above threshold")
            continue
        
        print(f"  ✓ Retrieved {len(chunks)} chunks")
        print()
        
        # Analyze individual chunks
        total_chars = 0
        total_tokens = 0
        
        for i, chunk in enumerate(chunks, 1):
            chars = len(chunk['text'])
            tokens = estimate_tokens(chunk['text'])
            total_chars += chars
            total_tokens += tokens
            
            print(f"  Chunk {i:2d}: {chars:6,} chars | ~{tokens:5,} tokens | sim={chunk['similarity']:.3f}")
            print(f"           Section: {chunk['section'][:60]}")
            print(f"           Page: {chunk['page']}")
        
        # Batch analysis
        separator = "\n\n---CHUNK SEPARATOR---\n\n"
        separator_overhead = len(separator) * (len(chunks) - 1)
        chunk_labels_overhead = sum(len(f"[Chunk {i+1}]\n") for i in range(len(chunks)))
        
        batch_chars = total_chars + separator_overhead + chunk_labels_overhead
        batch_tokens = estimate_tokens(str(batch_chars))
        
        print()
        print(f"  {'─'*76}")
        print(f"  BATCH SUMMARY:")
        print(f"  {'─'*76}")
        print(f"  Total content:        {total_chars:8,} chars | ~{total_tokens:6,} tokens")
        print(f"  Separators overhead:  {separator_overhead:8,} chars")
        print(f"  Labels overhead:      {chunk_labels_overhead:8,} chars")
        print(f"  {'─'*76}")
        print(f"  TOTAL BATCH SIZE:     {batch_chars:8,} chars | ~{batch_tokens:6,} tokens")
        print(f"  {'─'*76}")
        
        # Add prompt template overhead estimate
        prompt_template_overhead = 500  # Rough estimate for the prompt template
        total_with_prompt = batch_tokens + prompt_template_overhead
        
        print(f"  With prompt template: ~{total_with_prompt:6,} tokens")
        print()
        
        # Safety assessment
        print(f"  SAFETY ASSESSMENT:")
        if total_with_prompt < 2000:
            status = "✓ SAFE"
            color = "green"
        elif total_with_prompt < 4000:
            status = "⚠ MODERATE"
            color = "yellow"
        elif total_with_prompt < 8000:
            status = "⚠ HIGH"
            color = "orange"
        else:
            status = "✗ UNSAFE"
            color = "red"
        
        print(f"  {status} - {total_with_prompt:,} tokens")
        print(f"  Model context: Most models support 4K-8K input tokens")
        print(f"  Bedrock models typically: 4K-200K depending on model")
        
        all_batches.append({
            'keyword': kw,
            'num_chunks': len(chunks),
            'total_chars': batch_chars,
            'estimated_tokens': total_with_prompt,
            'status': status
        })
    
    # Overall summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    
    for batch in all_batches:
        print(f"  {batch['keyword']:20s}: {batch['num_chunks']:2d} chunks | "
              f"{batch['estimated_tokens']:6,} tokens | {batch['status']}")
    
    print()
    print(f"RECOMMENDATIONS:")
    print(f"  • If using Claude 3.5 Sonnet: 200K context - all batches are safe")
    print(f"  • If using older models (4K-8K): consider splitting batches")
    print(f"  • Current MAX_CHUNKS_PER_LLM_BATCH in code: check llm_extractor.py")
    print(f"  • Consider processing chunks individually if >8K tokens per batch")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze LLM batch sizes")
    parser.add_argument("--collection", default="financial_docs", help="Qdrant collection")
    parser.add_argument("--keywords", nargs="+", default=["net income", "revenue"], help="Keywords to test")
    parser.add_argument("--n-chunks", type=int, default=10, help="Chunks per keyword")
    parser.add_argument("--threshold", type=float, default=0.15, help="Similarity threshold")
    
    args = parser.parse_args()
    
    analyze_batch_size(
        collection_name=args.collection,
        keywords=args.keywords,
        n_chunks=args.n_chunks,
        threshold=args.threshold
    )
