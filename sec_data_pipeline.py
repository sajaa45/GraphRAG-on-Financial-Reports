"""
SEC Financial Data Pipeline
Efficiently processes SEC financial statement data with minimal resource usage.
Supports downloading and filtering data by year/quarter.
"""

import pandas as pd
import requests
import zipfile
import io
import os
from pathlib import Path
from typing import List, Optional, Dict
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SECDataPipeline:
    """Pipeline for downloading and processing SEC financial data efficiently."""
    
    BASE_URL = "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets/"
    
    def __init__(self, output_dir: str = "data", chunk_size: int = 50000):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to save processed data
            chunk_size: Number of rows to process at once (for memory efficiency)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        
    def download_quarter_data(self, year: int, quarter: int) -> Dict[str, Path]:
        """
        Download SEC data for a specific quarter.
        
        Args:
            year: Year (e.g., 2024)
            quarter: Quarter (1-4)
            
        Returns:
            Dictionary with paths to downloaded TSV files
        """
        # SEC changed format: now uses "2024q1" for older data, "2024 Q1" for newer
        # Try both formats
        quarter_str_old = f"{year}q{quarter}"
        quarter_str_new = f"{year}%20Q{quarter}"  # URL encoded space
        
        # Try new format first (2020+)
        zip_url = f"{self.BASE_URL}/{quarter_str_new}.zip"
        
        logger.info(f"Downloading data for {year} Q{quarter}...")
        
        headers = {
            'User-Agent': "User (your_email@example.com)"
        }
        
        # Try new format first
        try:
            response = requests.get(zip_url, headers=headers, timeout=300)
            response.raise_for_status()
            quarter_str = f"{year}q{quarter}"
        except requests.exceptions.RequestException:
            # Try old format
            logger.info(f"Trying alternate URL format...")
            zip_url = f"{self.BASE_URL}/{quarter_str_old}_notes.zip"
            try:
                response = requests.get(zip_url, headers=headers, timeout=300)
                response.raise_for_status()
                quarter_str = quarter_str_old
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {year} Q{quarter}: {e}")
                raise
        
        # Extract zip file
        temp_dir = self.output_dir / "temp" / quarter_str
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(temp_dir)
        
        logger.info(f"Downloaded and extracted {year} Q{quarter}")
        
        # Return paths to key files
        return {
            'num': temp_dir / 'num.txt',
            'sub': temp_dir / 'sub.txt',
            'tag': temp_dir / 'tag.txt'
        }
    
    def filter_data_chunked(
        self,
        num_path: Path,
        sub_path: Path,
        output_path: Path,
        form_types: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None
    ) -> None:
        """
        Filter and merge data using chunked processing for memory efficiency.
        
        Args:
            num_path: Path to num.txt file
            sub_path: Path to sub.txt file
            output_path: Path to save filtered output
            form_types: List of form types to include (e.g., ['10-K', '10-Q'])
            tags_filter: List of specific tags to include (optional)
        """
        logger.info("Starting chunked data filtering...")
        
        # Default to annual reports if not specified
        if form_types is None:
            form_types = ['10-K']
        
        # Step 1: Load and filter submissions (usually smaller)
        logger.info("Loading submissions data...")
        sub_df = pd.read_csv(
            sub_path,
            sep='\t',
            low_memory=False,
            keep_default_na=False,
            usecols=['adsh', 'cik', 'name', 'sic', 'form']
        )
        
        # Filter by form type
        sub_filtered = sub_df[sub_df['form'].isin(form_types)].copy()
        logger.info(f"Filtered submissions: {len(sub_filtered)} records")
        
        # Create a set of valid adsh for faster lookup
        valid_adsh = set(sub_filtered['adsh'])
        
        # Step 2: Process numeric facts in chunks
        logger.info("Processing numeric facts in chunks...")
        first_chunk = True
        total_rows = 0
        
        # Columns to keep from num.txt
        num_cols = ['adsh', 'tag', 'ddate', 'qtrs', 'value', 'footnote']
        
        for chunk in pd.read_csv(
            num_path,
            sep='\t',
            low_memory=False,
            keep_default_na=False,
            usecols=num_cols,
            chunksize=self.chunk_size
        ):
            # Filter by valid adsh
            chunk_filtered = chunk[chunk['adsh'].isin(valid_adsh)]
            
            # Optional: Filter by specific tags
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
                    output_path,
                    mode='w' if first_chunk else 'a',
                    header=first_chunk,
                    index=False
                )
                
                first_chunk = False
                total_rows += len(merged)
                logger.info(f"Processed chunk: {total_rows} total rows")
        
        logger.info(f"Filtering complete. Total rows: {total_rows}")
    
    def process_quarter(
        self,
        year: int,
        quarter: int,
        form_types: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None,
        keep_temp: bool = False
    ) -> Path:
        """
        Download and process data for a specific quarter.
        
        Args:
            year: Year
            quarter: Quarter (1-4)
            form_types: Form types to include
            tags_filter: Specific tags to include
            keep_temp: Whether to keep temporary downloaded files
            
        Returns:
            Path to filtered output file
        """
        quarter_str = f"{year}q{quarter}"
        output_path = self.output_dir / f"filtered_{quarter_str}.csv"
        
        # Check if already processed
        if output_path.exists():
            logger.info(f"Output file already exists: {output_path}")
            return output_path
        
        # Download data
        file_paths = self.download_quarter_data(year, quarter)
        
        # Filter data
        self.filter_data_chunked(
            num_path=file_paths['num'],
            sub_path=file_paths['sub'],
            output_path=output_path,
            form_types=form_types,
            tags_filter=tags_filter
        )
        
        # Cleanup temp files
        if not keep_temp:
            temp_dir = self.output_dir / "temp" / quarter_str
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary files for {quarter_str}")
        
        return output_path
    
    def process_year(
        self,
        year: int,
        form_types: Optional[List[str]] = None,
        tags_filter: Optional[List[str]] = None,
        combine: bool = True
    ) -> List[Path]:
        """
        Process all quarters for a given year.
        
        Args:
            year: Year to process
            form_types: Form types to include
            tags_filter: Specific tags to include
            combine: Whether to combine all quarters into one file
            
        Returns:
            List of output file paths
        """
        output_files = []
        
        for quarter in range(1, 5):
            try:
                output_path = self.process_quarter(
                    year=year,
                    quarter=quarter,
                    form_types=form_types,
                    tags_filter=tags_filter
                )
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {year}q{quarter}: {e}")
                continue
        
        # Optionally combine all quarters
        if combine and len(output_files) > 0:
            combined_path = self.output_dir / f"filtered_{year}_combined.csv"
            logger.info(f"Combining {len(output_files)} files...")
            
            first = True
            for file_path in output_files:
                df = pd.read_csv(file_path)
                df.to_csv(
                    combined_path,
                    mode='w' if first else 'a',
                    header=first,
                    index=False
                )
                first = False
            
            logger.info(f"Combined file saved: {combined_path}")
            return [combined_path]
        
        return output_files


def main():
    """Example usage of the pipeline."""
    pipeline = SECDataPipeline(output_dir="data", chunk_size=50000)
    
    # Example 1: Process a specific quarter
    # pipeline.process_quarter(year=2025, quarter=1, form_types=['10-K'])
    
    # Example 2: Process entire year
    # pipeline.process_year(year=2025, form_types=['10-K', '10-Q'])
    
    # Example 3: Process with specific tags
    # tags_of_interest = [
    #     'Assets',
    #     'Liabilities',
    #     'StockholdersEquity',
    #     'Revenues',
    #     'NetIncomeLoss'
    # ]
    # pipeline.process_quarter(
    #     year=2025,
    #     quarter=1,
    #     form_types=['10-K'],
    #     tags_filter=tags_of_interest
    # )
    
    print("Pipeline ready. Uncomment examples in main() to run.")


if __name__ == "__main__":
    main()
