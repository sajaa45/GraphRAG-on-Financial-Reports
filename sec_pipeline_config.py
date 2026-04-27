"""
Configuration file for SEC Data Pipeline
Customize these settings based on your needs
"""

# Years and quarters to process
YEARS_TO_PROCESS = [2024]  # 2024 has all quarters available  

# Form types to include
FORM_TYPES = [
    '10-K',    # Annual reports
    # '10-Q',  # Quarterly reports
    # '8-K',   # Current reports
]

# Specific financial tags to filter (None = include all)
# This can significantly reduce file size and processing time

# Example: Only include key financial metrics
TAGS_FILTER = [
    'Assets',
    'AssetsCurrent',
    'Liabilities',
    'LiabilitiesCurrent',
    'StockholdersEquity',
    'Revenues',
    'RevenueFromContractWithCustomerExcludingAssessedTax',
    'CostOfRevenue',
    'GrossProfit',
    'OperatingIncomeLoss',
    'NetIncomeLoss',
    'EarningsPerShareBasic',
    'EarningsPerShareDiluted',
    'CashAndCashEquivalentsAtCarryingValue',
    'AccountsReceivableNetCurrent',
    'InventoryNet',
    'PropertyPlantAndEquipmentNet',
    'Goodwill',
    'IntangibleAssetsNetExcludingGoodwill',
    'AccountsPayableCurrent',
    'LongTermDebt',
    'CommonStockSharesOutstanding',
]

# Processing settings
CHUNK_SIZE = 50000  # Rows to process at once (adjust based on available RAM)
OUTPUT_DIR = "data"  # Where to save processed files
KEEP_TEMP_FILES = False  # Keep downloaded raw files
COMBINE_QUARTERS = True  # Combine all quarters into one file per year

# Industry filters (SIC codes)
# Set to None to include all industries

# Example: Only financial services
SIC_FILTER = [
'6000',  # Depository Institutions
#     '6199',  # Finance Services
#     '6200',  # Security & Commodity Brokers
 ]

# Company filters (CIK codes)
# Set to None to include all companies
CIK_FILTER = None

# Example: Specific companies
# CIK_FILTER = [
#     '0000320193',  # Apple Inc.
#     '0001018724',  # Amazon.com Inc.
#     '0001652044',  # Alphabet Inc.
# ]
