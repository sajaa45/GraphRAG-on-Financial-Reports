// Run these commands in Neo4j Browser or cypher-shell as admin

// Grant all privileges on default database to Kimco user
GRANT ALL ON DATABASE neo4j TO Kimco;

// Alternatively, grant specific privileges:
// GRANT MATCH, CREATE, DELETE, SET_LABEL, SET_PROPERTY, REMOVE_LABEL ON GRAPH * TO Kimco;
