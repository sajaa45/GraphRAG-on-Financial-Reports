"""
Run the SEC Data Pipeline with configuration
"""

import sys
from pathlib import Path
from sec_data_pipeline import SECDataPipeline
import sec_pipeline_config as config
import logging

logger = logging.getLogger(__name__)


def apply_additional_filters(input_path: Path, output_path: Path) -> None:
    """
    Apply additional filters like SIC or CIK codes.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save filtered output
    """
    import pandas as pd
    
    logger.info(f"Applying additional filters to {input_path}...")
    
    # Read in chunks to handle large files
    first_chunk = True
    total_rows = 0
    
    for chunk in pd.read_csv(input_path, chunksize=config.CHUNK_SIZE):
        # Apply SIC filter
        if config.SIC_FILTER is not None:
            chunk = chunk[chunk['sic'].astype(str).isin(
                [str(sic) for sic in config.SIC_FILTER]
            )]
        
        # Apply CIK filter
        if config.CIK_FILTER is not None:
            chunk = chunk[chunk['cik'].astype(str).isin(
                [str(cik) for cik in config.CIK_FILTER]
            )]
        
        if len(chunk) > 0:
            chunk.to_csv(
                output_path,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            first_chunk = False
            total_rows += len(chunk)
    
    logger.info(f"Additional filtering complete. Total rows: {total_rows}")


def main():
    """Run the pipeline with configuration."""
    
    # Initialize pipeline
    pipeline = SECDataPipeline(
        output_dir=config.OUTPUT_DIR,
        chunk_size=config.CHUNK_SIZE
    )
    
    all_output_files = []
    
    # Process each year
    for year in config.YEARS_TO_PROCESS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing year: {year}")
        logger.info(f"{'='*60}\n")
        
        try:
            output_files = pipeline.process_year(
                year=year,
                form_types=config.FORM_TYPES,
                tags_filter=config.TAGS_FILTER,
                combine=config.COMBINE_QUARTERS
            )
            
            # Apply additional filters if needed
            if config.SIC_FILTER or config.CIK_FILTER:
                for file_path in output_files:
                    filtered_path = file_path.parent / f"{file_path.stem}_filtered_final.csv"
                    apply_additional_filters(file_path, filtered_path)
                    all_output_files.append(filtered_path)
            else:
                all_output_files.extend(output_files)
                
        except Exception as e:
            logger.error(f"Failed to process year {year}: {e}")
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Pipeline Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Processed {len(config.YEARS_TO_PROCESS)} years")
    logger.info(f"Generated {len(all_output_files)} output files:")
    for file_path in all_output_files:
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  - {file_path.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
