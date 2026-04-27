"""
Process local SEC data files (TSV format)
Use this when you already have downloaded SEC data files locally
"""

import pandas as pd
from pathlib import Path
import logging
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_local_files(
    num_file: str,
    sub_file: str,
    output_file: str,
    form_types: Optional[List[str]] = None,
    tags_filter: Optional[List[str]] = None,
    sic_filter: Optional[List[str]] = None,
    chunk_size: int = 50000
) -> None:
    """
    Process local SEC TSV files with memory-efficient chunking.
    
    Args:
        num_file: Path to num.tsv or num.txt file
        sub_file: Path to sub.tsv or sub.txt file
        output_file: Path to save filtered CSV
        form_types: List of form types to include (e.g., ['10-K'])
        tags_filter: List of specific tags to include
        sic_filter: List of SIC codes to include
        chunk_size: Number of rows to process at once
    """
    
    # Default to annual reports
    if form_types is None:
        form_types = ['10-K']
    
    logger.info("Loading submissions data...")
    
    # Load submissions with only needed columns
    sub_cols = ['adsh', 'cik', 'name', 'sic', 'form']
    sub_df = pd.read_csv(
        sub_file,
        sep='\t',
        low_memory=False,
        keep_default_na=False,
        usecols=sub_cols
    )
    
    logger.info(f"Total submissions: {len(sub_df)}")
    
    # Filter by form type
    sub_filtered = sub_df[sub_df['form'].isin(form_types)].copy()
    logger.info(f"After form filter: {len(sub_filtered)} records")
    
    # Filter by SIC if specified
    if sic_filter:
        sub_filtered = sub_filtered[
            sub_filtered['sic'].astype(str).isin([str(s) for s in sic_filter])
        ]
        logger.info(f"After SIC filter: {len(sub_filtered)} records")
    
    # Create set of valid adsh for faster lookup
    valid_adsh = set(sub_filtered['adsh'])
    logger.info(f"Valid ADSH count: {len(valid_adsh)}")
    
    # Process numeric facts in chunks
    logger.info("Processing numeric facts in chunks...")
    num_cols = ['adsh', 'tag', 'ddate', 'qtrs', 'value', 'footnote']
    
    first_chunk = True
    total_rows = 0
    chunks_processed = 0
    
    for chunk in pd.read_csv(
        num_file,
        sep='\t',
        low_memory=False,
        keep_default_na=False,
        usecols=num_cols,
        chunksize=chunk_size
    ):
        chunks_processed += 1
        
        # Filter by valid adsh
        chunk_filtered = chunk[chunk['adsh'].isin(valid_adsh)]
        
        # Filter by tags if specified
        if tags_filter:
            chunk_filtered = chunk_filtered[
                chunk_filtered['tag'].isin(tags_filter)
            ]
        
        if len(chunk_filtered) > 0:
            # Merge with submission data
            merged = chunk_filtered.merge(
                sub_filtered,
                on='adsh',
                how='left'
            )
            
            # Write to output
            merged.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            
            first_chunk = False
            total_rows += len(merged)
        
        if chunks_processed % 10 == 0:
            logger.info(f"Processed {chunks_processed} chunks, {total_rows} rows written")
    
    logger.info(f"Complete! Total rows: {total_rows}")
    logger.info(f"Output saved to: {output_file}")
    
    # Show file size
    output_path = Path(output_file)
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Output file size: {size_mb:.2f} MB")


def main():
    """Example usage."""
    
    # Example 1: Process files from a specific directory
    # Adjust paths to match your file locations
    process_local_files(
        num_file='/path/to/num.tsv',  # Update this path
        sub_file='/path/to/sub.tsv',  # Update this path
        output_file='data/filtered_data.csv',
        form_types=['10-K'],
        chunk_size=50000
    )
    
    # Example 2: With tag filtering for specific metrics
    # key_tags = [
    #     'Assets',
    #     'Liabilities',
    #     'StockholdersEquity',
    #     'Revenues',
    #     'NetIncomeLoss',
    #     'CashAndCashEquivalentsAtCarryingValue'
    # ]
    # 
    # process_local_files(
    #     num_file='/path/to/num.tsv',
    #     sub_file='/path/to/sub.tsv',
    #     output_file='data/filtered_key_metrics.csv',
    #     form_types=['10-K', '10-Q'],
    #     tags_filter=key_tags,
    #     chunk_size=50000
    # )
    
    # Example 3: Filter by industry (SIC codes)
    # process_local_files(
    #     num_file='/path/to/num.tsv',
    #     sub_file='/path/to/sub.tsv',
    #     output_file='data/filtered_tech_companies.csv',
    #     form_types=['10-K'],
    #     sic_filter=['3674', '7370', '7371'],  # Tech industry codes
    #     chunk_size=50000
    # )


if __name__ == "__main__":
    # Update the paths in main() before running
    print("Update file paths in main() before running")
    # main()
