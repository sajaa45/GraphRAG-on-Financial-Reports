# Use a Python image with build tools already included
FROM python:3.11

WORKDIR /app

# No apt-get needed - python:3.11 already has build tools

RUN pip install --upgrade pip && \
    pip install "numpy>=1.26.0,<2.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy all application files
COPY sections_merging_pages.py .
COPY sections_parser_pdf.py .
COPY chunking.py .
COPY chunking_vectorestore_pipeline.py .
COPY domain_relation_extraction_config.py .
COPY domain_industry_node_to_sic.py .
COPY domain_multi_relation_kg_builder.py .
COPY lexical_wrapper_kg.py .
COPY lexical_kG_building.py .
COPY lexical_graphrag_system.py .
COPY peers_kg_builder.py .
COPY graphrag_credit_risk.py .
# Create directories for input, output, and samples
RUN mkdir -p /app/input /app/output /app/samples

ENV PYTHONUNBUFFERED=1

# Use the new pipeline script that handles JSON processing and chunking
CMD ["python", "chunking_vectorestore_pipeline.py"]