"""
Check the length of source_text in extraction results
This helps identify if risks have properly isolated source paragraphs or full batches
"""
import json
import os

def analyze_source_text_lengths(
    extraction_result_path: str = None,
    structured_risks_path: str = None
):
    """Analyze source_text lengths from both extraction results and structured risks."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if extraction_result_path is None:
        extraction_result_path = os.path.join(
            script_dir, '..', '..', 'retrival_results', 'extraction_result.json'
        )
    
    if structured_risks_path is None:
        structured_risks_path = os.path.join(
            script_dir, '..', '..', '..', 'peers_sec', 'FACES_RISK', 'structured_risks.json'
        )
    
    print("="*80)
    print("Source Text Length Analysis")
    print("="*80)
    
    # Analyze extraction results (what validation uses)
    print("\n[1] Analyzing extraction_result.json (used by validation)")
    print("-"*80)
    
    with open(extraction_result_path, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    
    extraction_lengths = []
    extraction_details = []
    
    for result in extraction_data.get('results', []):
        # Check peer risks
        if result.get('peer_risks'):
            peer_name = result.get('peer', 'Unknown')
            for risk in result['peer_risks']:
                source_text = risk.get('source_text', '')
                length = len(source_text)
                extraction_lengths.append(length)
                extraction_details.append({
                    'company': peer_name,
                    'risk_name': risk.get('name', 'Unknown'),
                    'length': length,
                    'source': 'peer'
                })
        
        # Check target risks
        if result.get('target_risks'):
            target_name = result.get('target', 'Unknown')
            for risk in result['target_risks']:
                metadata = risk.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                
                source_text = metadata.get('source_text', '')
                length = len(source_text)
                extraction_lengths.append(length)
                extraction_details.append({
                    'company': target_name,
                    'risk_name': risk.get('name', 'Unknown'),
                    'length': length,
                    'source': 'target'
                })
    
    if extraction_lengths:
        print(f"Total risks: {len(extraction_lengths)}")
        print(f"Min length: {min(extraction_lengths):,} chars")
        print(f"Max length: {max(extraction_lengths):,} chars")
        print(f"Average length: {sum(extraction_lengths)/len(extraction_lengths):,.0f} chars")
        print(f"Median length: {sorted(extraction_lengths)[len(extraction_lengths)//2]:,} chars")
        
        over_3000 = sum(1 for l in extraction_lengths if l > 3000)
        over_5000 = sum(1 for l in extraction_lengths if l > 5000)
        print(f"\nOver 3000 chars: {over_3000} ({over_3000/len(extraction_lengths)*100:.1f}%)")
        print(f"Over 5000 chars: {over_5000} ({over_5000/len(extraction_lengths)*100:.1f}%)")
        
        # Distribution
        print("\nLength Distribution:")
        bins = [
            (0, 500, "0-500"),
            (500, 1000, "500-1K"),
            (1000, 2000, "1K-2K"),
            (2000, 3000, "2K-3K"),
            (3000, 5000, "3K-5K"),
            (5000, 10000, "5K-10K"),
            (10000, float('inf'), "10K+"),
        ]
        
        for min_len, max_len, label in bins:
            count = sum(1 for l in extraction_lengths if min_len <= l < max_len)
            pct = count / len(extraction_lengths) * 100
            bar = "█" * int(pct / 2)
            print(f"  {label:>8}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Show longest
        print("\nTop 10 Longest source_text:")
        extraction_details.sort(key=lambda x: x['length'], reverse=True)
        for i, item in enumerate(extraction_details[:10], 1):
            print(f"  {i}. {item['company']}: {item['risk_name'][:50]}... ({item['length']:,} chars)")
    else:
        print("No risks found in extraction results")
    
    # Analyze structured risks (source data)
    print("\n[2] Analyzing structured_risks.json (source data)")
    print("-"*80)
    
    if os.path.exists(structured_risks_path):
        with open(structured_risks_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        structured_lengths = []
        structured_details = []
        
        for company in structured_data:
            company_name = company.get('company_name', 'Unknown')
            for risk in company.get('risks', []):
                metadata = risk.get('metadata', {})
                source_text = metadata.get('source_text', '')
                length = len(source_text)
                structured_lengths.append(length)
                structured_details.append({
                    'company': company_name,
                    'risk_name': risk.get('risk_name', 'Unknown'),
                    'length': length
                })
        
        if structured_lengths:
            print(f"Total risks: {len(structured_lengths)}")
            print(f"Min length: {min(structured_lengths):,} chars")
            print(f"Max length: {max(structured_lengths):,} chars")
            print(f"Average length: {sum(structured_lengths)/len(structured_lengths):,.0f} chars")
            print(f"Median length: {sorted(structured_lengths)[len(structured_lengths)//2]:,} chars")
            
            over_3000 = sum(1 for l in structured_lengths if l > 3000)
            over_5000 = sum(1 for l in structured_lengths if l > 5000)
            print(f"\nOver 3000 chars: {over_3000} ({over_3000/len(structured_lengths)*100:.1f}%)")
            print(f"Over 5000 chars: {over_5000} ({over_5000/len(structured_lengths)*100:.1f}%)")
            
            # Distribution
            print("\nLength Distribution:")
            for min_len, max_len, label in bins:
                count = sum(1 for l in structured_lengths if min_len <= l < max_len)
                pct = count / len(structured_lengths) * 100
                bar = "█" * int(pct / 2)
                print(f"  {label:>8}: {count:4d} ({pct:5.1f}%) {bar}")
            
            # Show longest
            print("\nTop 10 Longest source_text:")
            structured_details.sort(key=lambda x: x['length'], reverse=True)
            for i, item in enumerate(structured_details[:10], 1):
                print(f"  {i}. {item['company']}: {item['risk_name'][:50]}... ({item['length']:,} chars)")
        else:
            print("No risks found in structured risks")
    else:
        print(f"File not found: {structured_risks_path}")
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if extraction_lengths and structured_lengths:
        if max(extraction_lengths) > 3000 or max(structured_lengths) > 3000:
            print("⚠ WARNING: Some source_text entries exceed 3000 chars")
            print("  This indicates risks are still getting full batch text instead of individual paragraphs")
            print("  Action needed:")
            print("  1. Verify MAX_BATCH_PARAGRAPHS = 1 in process_risks_simple.py")
            print("  2. Re-run: python peers_sec/FACES_RISK/process_risks_simple.py")
            print("  3. Rebuild knowledge graph with new structured_risks.json")
            print("  4. Re-run validation")
        else:
            print("✓ All source_text entries are under 3000 chars")
            print("  Risks appear to have properly isolated source paragraphs")
    
    print("="*80)


if __name__ == "__main__":
    analyze_source_text_lengths()
