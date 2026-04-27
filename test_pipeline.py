"""
Quick test of the SEC pipeline with one quarter
"""

from sec_data_pipeline import SECDataPipeline
import logging

logging.basicConfig(level=logging.INFO)

# Test with just Q1 2024
pipeline = SECDataPipeline(output_dir="data", chunk_size=50000)

print("Testing download for 2024 Q1...")
try:
    output = pipeline.process_quarter(
        year=2024,
        quarter=1,
        form_types=['10-K'],
        tags_filter=None  # Include all tags for now
    )
    print(f"\nSuccess! Output saved to: {output}")
    
    # Check file size
    if output.exists():
        size_mb = output.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        
        # Show first few rows
        import pandas as pd
        df = pd.read_csv(output, nrows=5)
        print(f"\nFirst few rows:")
        print(df)
        print(f"\nTotal columns: {len(df.columns)}")
        print(f"Columns: {df.columns.tolist()}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
