"""
Ground Truth Validation for Risk Factor Chunks

For each risk chunk retrieved by the pipeline, uses an LLM-as-judge to verify
whether the chunk is semantically relevant to its assigned risk theme.

Output CSV columns:
{company_name, cik, risk_theme, chunk_text, is_semantically_relevant,
 relevance_explanation, document_url, source, notes}
"""

import json
import os
import re
import time
from typing import List, Tuple
import csv
from dotenv import load_dotenv
import boto3

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '.env'))

# Max characters per chunk to avoid exceeding LLM context limits
# Qwen 3 Next 80B has 32K token context (~128K chars), so we can handle longer chunks
# Set to 15K to be conservative while allowing full risk paragraphs
MAX_CHUNK_CHARS = 15000


def _best_chunk_text(*candidates: str) -> str:
    """Return the first candidate with meaningful content (>10 words)."""
    for text in candidates:
        if text and len(text.split()) > 10:
            return text.strip()
    return ""


# Max (chunk, theme) evaluations per LLM call — keep small for reliable parsing.
MAX_EVALS_PER_CALL = 10


def check_chunks_against_themes(
    chunk_theme_groups: List[Tuple[str, List[str]]],  # (chunk_text, [risk_theme, ...])
    bedrock_client,
    model_id: str,
) -> List[List[Tuple[bool, str]]]:
    """
    Evaluate each chunk against all of its associated risk themes in one LLM call.

    chunk_theme_groups: list of (chunk_text, [theme1, theme2, ...])
    Returns a parallel list of lists: results[chunk_idx][theme_idx] = (is_relevant, explanation)
    Falls back to (False, error_msg) for any item that cannot be parsed.
    """
    prompt_body = ""
    total_evaluations = 0
    for ci, (chunk_text, themes) in enumerate(chunk_theme_groups, 1):
        # Chunks should already be validated for length, but double-check
        if len(chunk_text) > MAX_CHUNK_CHARS:
            raise ValueError(f"Chunk {ci} exceeds {MAX_CHUNK_CHARS} chars ({len(chunk_text)} chars). This should have been caught earlier.")
            
        prompt_body += f"\n--- Chunk {ci} ---\nText:\n{chunk_text}\n\nEvaluate for these risk themes:\n"
        for ti, theme in enumerate(themes, 1):
            prompt_body += f"  {ci}.{ti}: {theme}\n"
        prompt_body += "\n"
        total_evaluations += len(themes)

    prompt = (
        "You are evaluating whether text chunks from 10-K filings discuss specific risk themes.\n\n"
        "For EACH labeled evaluation below, respond with exactly one line:\n"
        "  <chunk>.<theme>: YES|NO — <one-sentence explanation>\n\n"
        "Rules:\n"
        "- YES if the chunk clearly discusses that risk theme.\n"
        "- NO if the chunk does not discuss it or is only tangentially related.\n"
        "- Output ONLY the labeled lines — no extra text.\n\n"
        f"{prompt_body}"
    )

    max_tokens = max(512, total_evaluations * 220)

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
        )
        response_text = response['output']['message']['content'][0]['text'].strip()
        
        # Debug: Print first response to see format
        if total_evaluations <= 3:  # Only for small batches
            print(f"  [DEBUG] LLM Response:\n{response_text[:500]}")
            
    except (KeyError, IndexError, TypeError) as e:
        print(f"  ✗ Unexpected response structure: {e}")
        return [[(False, f"Error: {e}")] * len(themes) for _, themes in chunk_theme_groups]
    except Exception as e:
        print(f"  ✗ LLM call failed: {e}")
        return [[(False, f"Error: {e}")] * len(themes) for _, themes in chunk_theme_groups]

    # Initialize with default fallback
    results: List[List] = [[None] * len(themes) for _, themes in chunk_theme_groups]

    # Parse "C.T: YES|NO — explanation" lines
    for line in response_text.splitlines():
        line = line.strip()
        # Try multiple patterns to be more flexible
        # Pattern 1: "1.1: YES — explanation" or "Chunk 1.1: YES — explanation"
        m = re.match(r'^(?:Chunk\s*)?(\d+)\.(\d+)(?:\.\d+)?\s*[:\.\)]\s*(YES|NO)\s*[-—–]?\s*(.*)', line, re.IGNORECASE)
        if not m:
            # Pattern 2: "<chunk>.1.1: YES" (angle brackets)
            m = re.match(r'^<chunk>\.(\d+)\.(\d+)(?:\.\d+)?\s*[:\.\)]\s*(YES|NO)\s*[-—–]?\s*(.*)', line, re.IGNORECASE)
        if not m:
            # Pattern 3: "1.1 YES explanation" (no colon or dash)
            m = re.match(r'^(?:Chunk\s*)?(\d+)\.(\d+)(?:\.\d+)?\s+(YES|NO)\s+(.*)', line, re.IGNORECASE)
        if not m:
            # Pattern 4: Just look for numbers followed by YES/NO anywhere
            m = re.match(r'^(?:Chunk\s*)?(\d+)\.(\d+)(?:\.\d+)?.*?(YES|NO)(?:\s*[-—–:]\s*|\s+)(.*)', line, re.IGNORECASE)
        
        if not m:
            # Pattern 5: "1-1: YES — explanation" (dash separator between chunk and theme)
            m = re.match(r'^(?:Chunk\s*)?(\d+)-(\d+)\s*[:\.\)]\s*(YES|NO)\s*[-—–]?\s*(.*)', line, re.IGNORECASE)
        if not m:
            # Pattern 6: markdown bold "**1.1**: YES — explanation"
            m = re.match(r'^\*{0,2}(\d+)\.(\d+)\*{0,2}\s*[:\.\)]\s*(YES|NO)\s*[-—–]?\s*(.*)', line, re.IGNORECASE)

        if m:
            ci, ti = int(m.group(1)) - 1, int(m.group(2)) - 1
            if 0 <= ci < len(results) and 0 <= ti < len(results[ci]):
                verdict = m.group(3).upper() == "YES"
                explanation = m.group(4).strip() if len(m.groups()) >= 4 else "No explanation provided"
                results[ci][ti] = (verdict, explanation)

    failed_slots = [(ci, ti) for ci in range(len(results)) for ti in range(len(results[ci])) if results[ci][ti] is None]
    if failed_slots:
        print(f"  [DEBUG] Parse failures for slots: {failed_slots}")
        print(f"  [DEBUG] Full LLM response:\n{response_text}\n  [END DEBUG]")

    for ci in range(len(results)):
        for ti in range(len(results[ci])):
            if results[ci][ti] is None:
                results[ci][ti] = (False, "Could not parse LLM response for this item")

    parsed_count = sum(1 for ci in range(len(results)) for ti in range(len(results[ci]))
                      if results[ci][ti][1] != "Could not parse LLM response for this item")
    print(f"    Parsed {parsed_count}/{total_evaluations} evaluations successfully")

    return results


def validate_risk_chunks(
    extraction_result_path: str,
    output_csv_path: str,
    use_llm_judge: bool = True
):
    # Set up logging to both console and file
    log_file_path = output_csv_path.replace('.csv', '_log.txt')
    
    class TeeLogger:
        """Write to both console and file"""
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
        
        def close(self):
            self.log.close()
    
    import sys
    original_stdout = sys.stdout
    logger = TeeLogger(log_file_path)
    sys.stdout = logger
    
    try:
        _run_validation(extraction_result_path, output_csv_path, use_llm_judge)
    finally:
        sys.stdout = original_stdout
        logger.close()
        print(f"\n✓ Log saved to {log_file_path}")


def _run_validation(
    extraction_result_path: str,
    output_csv_path: str,
    use_llm_judge: bool = True
):
    print("="*70)
    print("Risk Factor Chunks — Semantic Relevance Validation")
    print("="*70)

    print(f"\n[1/4] Loading extraction results")
    with open(extraction_result_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)

    # Initialize AWS Bedrock client
    bedrock_client = None
    bedrock_model = None
    if use_llm_judge:
        try:
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            bedrock_model = os.getenv("BEDROCK_MODEL", "us.meta.llama3-2-90b-instruct-v1:0")
            bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=aws_region
            )
            print(f"  ✓ Bedrock client initialized (region: {aws_region}, model: {bedrock_model})")
        except Exception as e:
            print(f"  ⚠ Failed to initialize Bedrock client: {e} — skipping LLM checks")
            use_llm_judge = False

    # Collect chunks
    chunks_to_validate = []

    for result in extraction_data.get('results', []):
        if result.get('peer_risks'):
            peer_name = result.get('peer', 'Unknown')
            for risk_chunk in result['peer_risks']:
                cik = (risk_chunk.get('risk_id') or 'Unknown').split('_')[0]
                chunk_text = _best_chunk_text(
                    risk_chunk.get('source_text', ''),
                    risk_chunk.get('description', ''),
                    risk_chunk.get('why', ''),
                )
                chunks_to_validate.append({
                    'company_name': peer_name,
                    'cik': cik,
                    'risk_theme': risk_chunk.get('name', 'Unknown'),
                    'chunk_text': chunk_text,
                    'document_url': risk_chunk.get('document_url', ''),
                    'source': 'peer',
                })

        if result.get('target_risks'):
            target_name = result.get('target', 'Unknown')
            for risk_chunk in result['target_risks']:
                metadata = risk_chunk.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                chunk_text = _best_chunk_text(
                    metadata.get('source_text', ''),
                    risk_chunk.get('description', ''),
                    risk_chunk.get('why', ''),
                )
                chunks_to_validate.append({
                    'company_name': target_name,
                    'cik': 'target',
                    'risk_theme': risk_chunk.get('name', 'Unknown'),
                    'chunk_text': chunk_text,
                    'document_url': metadata.get('document_url', ''),
                    'source': 'target',
                })

    print(f"  → Found {len(chunks_to_validate)} risk chunks to validate")

    if not chunks_to_validate:
        print("\n✗ No risk chunks found in extraction results")
        return

    results_table = [
        {
            **chunk, 
            'chunk_length': len(chunk['chunk_text']) if chunk['chunk_text'] else 0,
            'is_semantically_relevant': 'N/A', 
            'relevance_explanation': '', 
            'notes': ''
        }
        for chunk in chunks_to_validate
    ]

    # --- Grouped LLM semantic relevance ---
    if use_llm_judge and bedrock_client:
        eligible, skipped = [], []
        too_long_chunks = []
        
        for row_idx, row in enumerate(results_table):
            chunk_text = row['chunk_text']
            
            # Check if chunk is too long
            if chunk_text and len(chunk_text) > MAX_CHUNK_CHARS:
                too_long_chunks.append({
                    'row_idx': row_idx,
                    'company': row['company_name'],
                    'theme': row['risk_theme'],
                    'length': len(chunk_text)
                })
                skipped.append(row_idx)
                results_table[row_idx]['is_semantically_relevant'] = False
                results_table[row_idx]['relevance_explanation'] = f'Chunk too long ({len(chunk_text)} chars, max {MAX_CHUNK_CHARS})'
            elif not chunk_text or len(chunk_text.split()) < 10:
                skipped.append(row_idx)
                results_table[row_idx]['is_semantically_relevant'] = False
                results_table[row_idx]['relevance_explanation'] = 'Chunk text missing or too short'
            else:
                eligible.append((row_idx, row))
        
        # Report chunks that are too long
        if too_long_chunks:
            print(f"\n  ⚠ WARNING: {len(too_long_chunks)} chunks exceed {MAX_CHUNK_CHARS} chars and will be skipped:")
            for item in too_long_chunks[:5]:  # Show first 5
                print(f"    - {item['company']}: {item['theme'][:50]}... ({item['length']} chars)")
            if len(too_long_chunks) > 5:
                print(f"    ... and {len(too_long_chunks) - 5} more")
            
            # Ask user if they want to continue
            print(f"\n  These chunks have source_text that is too long.")
            print(f"  This likely means the risk extraction didn't properly isolate individual paragraphs.")
            print(f"  You should re-run the extraction with the updated process_risks.py")
            print(f"\n  Continue validation anyway? (y/n): ", end="")
            
            # For non-interactive environments, default to continue
            try:
                import sys
                if sys.stdin.isatty():
                    response = input().strip().lower()
                    if response != 'y':
                        print("\n  Validation aborted. Please re-run risk extraction first.")
                        return
                else:
                    print("y (non-interactive mode)")
            except:
                print("y (input not available)")
        
        # Group row indices by chunk_text so repeated chunks are evaluated once.
        # chunk_groups: {chunk_text: [(row_idx, risk_theme), ...]}
        chunk_groups: dict = {}
        for row_idx, row in eligible:
            key = row['chunk_text']
            chunk_groups.setdefault(key, []).append((row_idx, row['risk_theme']))

        unique_chunks = list(chunk_groups.items())  # [(chunk_text, [(row_idx, theme), ...]), ...]
        total_evaluations = sum(len(rows) for _, rows in unique_chunks)
        dedup_savings = len(eligible) - len(unique_chunks)

        # Build calls capped at MAX_EVALS_PER_CALL evaluations each.
        # A single chunk with more themes than the cap is split across calls.
        calls: List[List[Tuple[str, list]]] = []  # each call: [(chunk_text, [(row_idx, theme)]), ...]
        current_call: list = []
        current_evals = 0
        for chunk_text, rows in unique_chunks:
            for i in range(0, len(rows), MAX_EVALS_PER_CALL):
                slice_ = rows[i:i + MAX_EVALS_PER_CALL]
                n = len(slice_)
                if current_evals + n > MAX_EVALS_PER_CALL and current_call:
                    calls.append(current_call)
                    current_call = []
                    current_evals = 0
                current_call.append((chunk_text, slice_))
                current_evals += n
        if current_call:
            calls.append(current_call)

        print(
            f"\n[2/4] LLM semantic relevance — {len(eligible)} chunks → "
            f"{len(unique_chunks)} unique texts ({dedup_savings} duplicates skipped), "
            f"{total_evaluations} theme evaluations in {len(calls)} calls of ≤{MAX_EVALS_PER_CALL}"
            f" ({len(skipped)} skipped — no usable text)"
        )

        for call_num, call in enumerate(calls, 1):
            total_evals_in_call = sum(len(rows) for _, rows in call)
            print(f"  Call [{call_num}/{len(calls)}] — {len(call)} chunk(s), {total_evals_in_call} evaluations...")

            llm_input = [(chunk_text, [theme for _, theme in rows]) for chunk_text, rows in call]
            call_results = check_chunks_against_themes(llm_input, bedrock_client, bedrock_model)

            for (chunk_text, rows), theme_verdicts in zip(call, call_results):
                for (row_idx, _), (is_relevant, explanation) in zip(rows, theme_verdicts):
                    results_table[row_idx]['is_semantically_relevant'] = is_relevant
                    results_table[row_idx]['relevance_explanation'] = explanation

            time.sleep(0.5)

    # Metrics
    print(f"\n[3/4] Metrics")
    total_chunks = len(results_table)
    print(f"  Total chunks: {total_chunks}")
    if use_llm_judge:
        relevant_chunks = sum(1 for r in results_table if r['is_semantically_relevant'] is True)
        irrelevant_chunks = sum(1 for r in results_table if r['is_semantically_relevant'] is False)
        semantic_precision = (relevant_chunks / total_chunks * 100) if total_chunks > 0 else 0
        print(f"  Relevant:   {relevant_chunks}")
        print(f"  Irrelevant: {irrelevant_chunks}")
        print(f"  Precision:  {semantic_precision:.2f}%")

    # Write CSV
    print(f"\n[4/4] Writing results to {output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'company_name', 'cik', 'risk_theme', 'chunk_length', 'chunk_text',
            'is_semantically_relevant', 'relevance_explanation',
            'document_url', 'source', 'notes',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)
    print(f"  ✓ Saved to {output_csv_path}")

    print("\n" + "="*70)
    print("Validation Complete")
    if use_llm_judge:
        print(f"Semantic Precision: {semantic_precision:.2f}%")
        
        # Analyze correlation between chunk length and relevance
        print("\n" + "-"*70)
        print("Chunk Length Analysis")
        print("-"*70)
        
        relevant_lengths = [r['chunk_length'] for r in results_table if r['is_semantically_relevant'] is True]
        irrelevant_lengths = [r['chunk_length'] for r in results_table if r['is_semantically_relevant'] is False]
        
        if relevant_lengths:
            print(f"Relevant chunks (n={len(relevant_lengths)}):")
            print(f"  Avg length: {sum(relevant_lengths)/len(relevant_lengths):,.0f} chars")
            print(f"  Min: {min(relevant_lengths):,} | Max: {max(relevant_lengths):,}")
        
        if irrelevant_lengths:
            print(f"\nIrrelevant chunks (n={len(irrelevant_lengths)}):")
            print(f"  Avg length: {sum(irrelevant_lengths)/len(irrelevant_lengths):,.0f} chars")
            print(f"  Min: {min(irrelevant_lengths):,} | Max: {max(irrelevant_lengths):,}")
        
        # Check for truncation patterns
        if irrelevant_lengths:
            very_long = sum(1 for l in irrelevant_lengths if l > 8000)
            if very_long > 0:
                print(f"\n⚠ {very_long} irrelevant chunks are >8000 chars")
                print(f"  This suggests chunks may contain multiple risks")
    
    print("="*70)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    extraction_result_path = os.path.join(
        script_dir, '..', '..', 'retrival_results', 'extraction_result.json'
    )
    output_csv_path = os.path.join(
        script_dir, '..', 'risks_validation_results.csv'
    )

    validate_risk_chunks(
        extraction_result_path,
        output_csv_path,
        use_llm_judge=True
    )
