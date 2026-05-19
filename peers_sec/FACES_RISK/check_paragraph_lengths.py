"""
Check the length of individual risk paragraphs in companies_risks.json
"""
import json
import os

def analyze_paragraph_lengths(input_file: str = "peers_sec/FACES_RISK/companies_risks.json"):
    """Analyze and report statistics on paragraph lengths."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        companies_data = json.load(f)
    
    print("="*80)
    print("Risk Paragraph Length Analysis")
    print("="*80)
    
    all_lengths = []
    company_stats = []
    
    for company in companies_data:
        cik = company["cik"]
        company_name = company.get("company_name", "Unknown")
        paragraphs = company["individual_risks"]
        
        # Calculate lengths
        lengths = [len(p) for p in paragraphs]
        all_lengths.extend(lengths)
        
        # Company statistics
        stats = {
            'cik': cik,
            'company_name': company_name,
            'total_paragraphs': len(paragraphs),
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'over_3000': sum(1 for l in lengths if l > 3000),
            'over_5000': sum(1 for l in lengths if l > 5000),
        }
        company_stats.append(stats)
        
        print(f"\n[{cik}] {company_name}")
        print(f"  Total paragraphs: {stats['total_paragraphs']}")
        print(f"  Length range: {stats['min_length']:,} - {stats['max_length']:,} chars")
        print(f"  Average length: {stats['avg_length']:,.0f} chars")
        print(f"  Over 3000 chars: {stats['over_3000']} ({stats['over_3000']/stats['total_paragraphs']*100:.1f}%)")
        print(f"  Over 5000 chars: {stats['over_5000']} ({stats['over_5000']/stats['total_paragraphs']*100:.1f}%)")
    
    # Overall statistics
    print("\n" + "="*80)
    print("Overall Statistics")
    print("="*80)
    
    total_paragraphs = sum(s['total_paragraphs'] for s in company_stats)
    total_over_3000 = sum(s['over_3000'] for s in company_stats)
    total_over_5000 = sum(s['over_5000'] for s in company_stats)
    
    print(f"Total companies: {len(companies_data)}")
    print(f"Total paragraphs: {total_paragraphs}")
    print(f"Min length: {min(all_lengths):,} chars")
    print(f"Max length: {max(all_lengths):,} chars")
    print(f"Average length: {sum(all_lengths)/len(all_lengths):,.0f} chars")
    print(f"Median length: {sorted(all_lengths)[len(all_lengths)//2]:,} chars")
    print(f"\nParagraphs over 3000 chars: {total_over_3000} ({total_over_3000/total_paragraphs*100:.1f}%)")
    print(f"Paragraphs over 5000 chars: {total_over_5000} ({total_over_5000/total_paragraphs*100:.1f}%)")
    
    # Length distribution
    print("\n" + "="*80)
    print("Length Distribution")
    print("="*80)
    
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
        count = sum(1 for l in all_lengths if min_len <= l < max_len)
        pct = count / len(all_lengths) * 100
        bar = "█" * int(pct / 2)
        print(f"{label:>8}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Find longest paragraphs
    print("\n" + "="*80)
    print("Top 10 Longest Paragraphs")
    print("="*80)
    
    longest_paragraphs = []
    for company in companies_data:
        for i, para in enumerate(company["individual_risks"]):
            longest_paragraphs.append({
                'company': company.get("company_name", "Unknown"),
                'cik': company["cik"],
                'index': i,
                'length': len(para),
                'preview': para[:100].replace('\n', ' ')
            })
    
    longest_paragraphs.sort(key=lambda x: x['length'], reverse=True)
    
    for i, para in enumerate(longest_paragraphs[:10], 1):
        print(f"\n{i}. {para['company']} (CIK {para['cik']}) - Paragraph {para['index']}")
        print(f"   Length: {para['length']:,} chars")
        print(f"   Preview: {para['preview']}...")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_paragraph_lengths()
