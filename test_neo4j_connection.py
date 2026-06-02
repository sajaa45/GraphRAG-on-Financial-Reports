"""Test Neo4j connection and permissions."""
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
user = os.getenv("NEO4J_USERNAME", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

print(f"Testing connection to {uri}")
print(f"Username: {user}")
print(f"Password: {'*' * len(password) if password else 'NOT SET'}")

try:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    print("✓ Driver created")
    
    with driver.session() as session:
        # Test read
        result = session.run("RETURN 1 as num")
        print(f"✓ Read test passed: {result.single()['num']}")
        
        # Test write (create a temporary node)
        result = session.run("CREATE (t:Test {name: 'test'}) RETURN t")
        print(f"✓ Write test passed: {result.single()['t']}")
        
        # Clean up
        session.run("MATCH (t:Test {name: 'test'}) DELETE t")
        print("✓ Delete test passed")
        
    driver.close()
    print("\n✓✓✓ All tests passed! Connection and permissions are working.")
    
except Exception as e:
    print(f"\n✗✗✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if Neo4j is running: http://localhost:7474")
    print("2. Verify credentials in .env file")
    print("3. Check user permissions in Neo4j")
