from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Initialize
model = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)
collection_name = "financial"

# Test different search queries
queries = [
    "item 1 business company overview",
    "item 1. business description",
    "risk factors",
    "financial performance results earnings statements",
]

print("=" * 100)
print("SECTION SIMILARITY ANALYSIS")
print("=" * 100)

for query in queries:
    print(f"\n\nQuery: '{query}'")
    print("-" * 100)
    
    # Embed the query
    query_embedding = model.encode(query).tolist()
    
    # Search for sections
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=Filter(must=[FieldCondition(key="type", match=MatchValue(value="section"))]),
        limit=5,
        with_payload=True
    ).points
    
    print(f"\n{'Rank':<6} {'Score':<8} {'Section Title':<80}")
    print("-" * 100)
    
    for i, hit in enumerate(results, 1):
        title = hit.payload.get('title', 'Unknown')
        score = hit.score
        print(f"{i:<6} {score:<8.4f} {title:<80}")
