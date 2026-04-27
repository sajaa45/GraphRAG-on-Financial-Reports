#!/usr/bin/env python3
"""
GraphRAG System using Neo4j Knowledge Graph and Local Ollama
Tests different retrieval approaches for query answering
"""

import os
import json
import time
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import requests

class GraphRAGSystem:
    def __init__(self, neo4j_uri: str, neo4j_username: str, neo4j_password: str, 
                 ollama_url: str = "http://localhost:11434", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize GraphRAG system
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            ollama_url: Ollama API URL
            embedding_model: Sentence transformer model for embeddings
        """
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        self.ollama_url = ollama_url
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Test connections
        self._test_neo4j_connection()
        self._test_ollama_connection()
        
        print(f"GraphRAG System initialized successfully")
        print(f"- Neo4j: {neo4j_uri}")
        print(f"- Ollama: {ollama_url}")
        print(f"- Embedding model: {embedding_model}")
    
    def _test_neo4j_connection(self):
        """Test Neo4j connection"""
        try:
            with self.neo4j_driver.session() as session:
                result = session.run("RETURN count(*) as total_nodes")
                total = result.single()['total_nodes']
                print(f"Neo4j connected: {total} total nodes")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def _test_ollama_connection(self):
        """Test Ollama connection and list available models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                print(f"Ollama connected: {len(models)} models available")
                print(f"Available models: {model_names}")
                return model_names
            else:
                raise ConnectionError(f"Ollama API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Failed to connect to Ollama. Make sure Ollama is running on localhost:11434")
    
    def get_available_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []
    
    def embed_query(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        return self.embedding_model.encode([query])[0]
    
    def rerank_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """
        Rerank chunks based on query type and content characteristics
        This addresses the limitation of pure semantic similarity
        """
        query_lower = query.lower()
        
        # Detect query type
        is_numeric_query = any(keyword in query_lower for keyword in [
            "how much", "how many", "net income", "revenue", "profit", "loss",
            "balance", "assets", "debt", "equity", "cash", "value", "amount",
            "total", "sum", "percentage", "%", "sar", "usd", "million", "billion"
        ])
        
        is_list_query = any(keyword in query_lower for keyword in [
            "what are", "list", "which", "name the", "identify", "enumerate",
            "types of", "categories", "examples of"
        ])
        
        is_definition_query = any(keyword in query_lower for keyword in [
            "what is", "define", "meaning of", "explain", "describe"
        ])
        
        is_comparison_query = any(keyword in query_lower for keyword in [
            "compare", "difference", "versus", "vs", "between", "higher", "lower"
        ])
        
        # Rerank each chunk
        for chunk in chunks:
            text = chunk['content']
            text_lower = text.lower()
            
            # Start with semantic similarity score
            score = chunk['similarity']
            
            # Boost for numeric queries if chunk contains numbers
            if is_numeric_query:
                # Check for actual numeric data (not just page numbers)
                has_numbers = any(char.isdigit() for char in text)
                has_currency = any(curr in text_lower for curr in ['sar', 'usd', 'million', 'billion'])
                has_percentage = '%' in text
                
                if has_numbers and (has_currency or has_percentage):
                    score += 0.25  # Strong boost for financial data
                elif has_numbers:
                    score += 0.15  # Moderate boost for any numbers
            
            # Boost for list queries if chunk contains list structures
            if is_list_query:
                list_indicators = [
                    text.count('\n1.'), text.count('\n2.'), text.count('\n3.'),
                    text.count('\n-'), text.count('\n•'), text.count('\n*'),
                    text.count('first'), text.count('second'), text.count('third')
                ]
                list_score = sum(1 for count in list_indicators if count > 0)
                
                if list_score >= 3:
                    score += 0.20  # Strong boost for clear lists
                elif list_score >= 1:
                    score += 0.10  # Moderate boost for some list structure
            
            # Boost for definition queries if chunk has definition patterns
            if is_definition_query:
                definition_patterns = [
                    ' is ', ' are ', ' means ', ' refers to ', ' defined as ',
                    'definition', 'meaning'
                ]
                if any(pattern in text_lower for pattern in definition_patterns):
                    score += 0.15
            
            # Boost for comparison queries if chunk has comparison language
            if is_comparison_query:
                comparison_patterns = [
                    'compared to', 'versus', 'higher than', 'lower than',
                    'increased', 'decreased', 'more than', 'less than'
                ]
                if any(pattern in text_lower for pattern in comparison_patterns):
                    score += 0.15
            
            # Penalize very short chunks (likely incomplete context)
            if len(text) < 100:
                score -= 0.10
            
            # Penalize chunks that are mostly tables without context
            if text.count('|') > len(text) / 20:  # More than 5% pipe characters
                score -= 0.05
            
            chunk['rerank_score'] = score
        
        # Sort by reranked score
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        print(f"  Reranked {len(chunks)} chunks based on query type")
        if is_numeric_query:
            print(f"    → Boosted chunks with financial data")
        if is_list_query:
            print(f"    → Boosted chunks with list structures")
        if is_definition_query:
            print(f"    → Boosted chunks with definitions")
        if is_comparison_query:
            print(f"    → Boosted chunks with comparisons")
        
        return reranked
    
    def find_similar_chunks(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Dict]:
        """
        Find most similar chunks using Neo4j and pre-computed embeddings stored in chunks
        Now retrieves top_k=15 by default for better recall
        """
        with self.neo4j_driver.session() as session:
            # Get all chunks with their content and pre-computed embeddings
            result = session.run("""
                MATCH (c:Chunk)
                RETURN c.chunk_id as chunk_id, 
                       c.content as content,
                       c.section_id as section_id,
                       c.page_range as page_range,
                       c.chunk_index_in_section as chunk_index,
                       c.total_chunks_in_section as total_chunks,
                       c.embedding as embedding
                ORDER BY c.chunk_id
            """)
            
            chunks = []
            chunk_embeddings = []
            
            for record in result:
                embedding = record['embedding']
                
                # Skip chunks without embeddings
                if not embedding or len(embedding) == 0:
                    continue
                
                chunk_data = {
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'section_id': record['section_id'],
                    'page_range': record['page_range'],
                    'chunk_index': record['chunk_index'],
                    'total_chunks': record['total_chunks']
                }
                chunks.append(chunk_data)
                chunk_embeddings.append(embedding)
            
            print(f"Loaded {len(chunks)} chunks with pre-computed embeddings for similarity search")
            
            if len(chunks) == 0:
                print("Warning: No chunks with embeddings found!")
                return []
            
            # Convert to numpy array and normalize for efficient computation
            chunk_embeddings = np.array(chunk_embeddings)
            
            # Normalize embeddings for stable cosine similarity
            chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            query_embedding_normalized = query_embedding / np.linalg.norm(query_embedding)
            
            # Calculate cosine similarities using normalized embeddings
            similarities = np.dot(chunk_embeddings, query_embedding_normalized)
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar_chunks = []
            for idx in top_indices:
                chunk = chunks[idx].copy()
                chunk['similarity'] = float(similarities[idx])
                similar_chunks.append(chunk)
            
            return similar_chunks
    
    def get_next_chunk(self, chunk_id: str) -> Dict:
        """Get the next chunk in sequence using Neo4j relationships"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c1:Chunk {chunk_id: $chunk_id})-[:NEXT_CHUNK]->(c2:Chunk)
                RETURN c2.chunk_id as chunk_id,
                       c2.content as content,
                       c2.section_id as section_id,
                       c2.page_range as page_range,
                       c2.chunk_index_in_section as chunk_index
            """, chunk_id=chunk_id)
            
            record = result.single()
            if record:
                return {
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'section_id': record['section_id'],
                    'page_range': record['page_range'],
                    'chunk_index': record['chunk_index']
                }
            return None
    
    def get_previous_chunk(self, chunk_id: str) -> Dict:
        """Get the previous chunk in sequence using Neo4j relationships"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (c1:Chunk {chunk_id: $chunk_id})-[:PREVIOUS_CHUNK]->(c2:Chunk)
                RETURN c2.chunk_id as chunk_id,
                       c2.content as content,
                       c2.section_id as section_id,
                       c2.page_range as page_range,
                       c2.chunk_index_in_section as chunk_index
            """, chunk_id=chunk_id)
            
            record = result.single()
            if record:
                return {
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'section_id': record['section_id'],
                    'page_range': record['page_range'],
                    'chunk_index': record['chunk_index']
                }
            return None
    
    def get_section_chunks(self, section_id: str) -> List[Dict]:
        """Get all chunks from the same section"""
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (s:Section {section_id: $section_id})-[:CONTAINS_CHUNK]->(c:Chunk)
                RETURN c.chunk_id as chunk_id,
                       c.content as content,
                       c.chunk_index_in_section as chunk_index,
                       c.page_range as page_range
                ORDER BY c.chunk_index_in_section
            """, section_id=section_id)
            
            chunks = []
            for record in result:
                chunks.append({
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'chunk_index': record['chunk_index'],
                    'page_range': record['page_range']
                })
            
            return chunks
    
    def query_ollama(self, prompt: str, model: str = "mistral:latest", max_tokens: int = 500) -> Dict:
        """Query Ollama model with improved error handling"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.1
                }
            }
            
            # Log prompt size for debugging
            prompt_size = len(prompt)
            if prompt_size > 10000:
                print(f"  ⚠ Large prompt: {prompt_size} chars")
            
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate", 
                json=payload,
                timeout=120  # 2 minute timeout
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", ""),
                    "model": model,
                    "response_time": round(end_time - start_time, 2),
                    "success": True,
                    "prompt_size": prompt_size
                }
            else:
                error_detail = ""
                try:
                    error_detail = response.json().get('error', '')
                except:
                    error_detail = response.text[:200]
                
                return {
                    "error": f"Ollama API error: {response.status_code} - {error_detail}",
                    "success": False,
                    "prompt_size": prompt_size
                }
        except requests.exceptions.Timeout:
            return {
                "error": "Ollama request timeout (>120s)",
                "success": False
            }
        except Exception as e:
            return {
                "error": f"Failed to query Ollama: {e}",
                "success": False
            }

    
    def approach_1_single_chunk(self, query: str, model: str = "mistral:latest") -> Dict:
        """Approach 1: Use only the most similar chunk"""
        print(f"\n=== Approach 1: Single Chunk Retrieval ===")
        
        # Find most similar chunk
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        print(f"Pages: {best_chunk['page_range']}")
        
        # Create prompt
        prompt = f"""Based on the following document excerpt, please answer the question.

Document excerpt:
{best_chunk['content']}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "single_chunk"
        result['chunks_used'] = 1
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = best_chunk['page_range']
        result['chunk_ids'] = [best_chunk['chunk_id']]
        
        return result
    
    def approach_2_sequential_chunks(self, query: str, model: str = "mistral:latest") -> Dict:
        """Approach 2: Use the most similar chunk + next chunk"""
        print(f"\n=== Approach 2: Sequential Chunks Retrieval ===")
        
        # Find most similar chunk
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        
        # Get next chunk
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = [best_chunk]
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found")
        
        # Combine content
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpts, please answer the question.

Document excerpts:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "sequential_chunks"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        
        return result
    

    
    def approach_4_context_window(self, query: str, model: str = "mistral:latest") -> Dict:
        """Approach 4: Use previous + current + next chunk (context window)"""
        print(f"\n=== Approach 4: Context Window Retrieval (Prev + Current + Next) ===")
        
        # Find most similar chunk
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=1)
        
        if not similar_chunks:
            return {"error": "No chunks found", "success": False}
        
        best_chunk = similar_chunks[0]
        print(f"Best match: {best_chunk['chunk_id']} (similarity: {best_chunk['similarity']:.3f})")
        
        # Get previous and next chunks
        previous_chunk = self.get_previous_chunk(best_chunk['chunk_id'])
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = []
        
        # Add previous chunk if available
        if previous_chunk:
            chunks_to_use.append(previous_chunk)
            print(f"Previous chunk: {previous_chunk['chunk_id']}")
        else:
            print("No previous chunk found")
        
        # Add the best matching chunk
        chunks_to_use.append(best_chunk)
        
        # Add next chunk if available
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found")
        
        print(f"Using {len(chunks_to_use)} chunks in context window")
        
        # Combine content in order
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpts (in sequential order), please answer the question.

Document excerpts:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "context_window"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        result['has_previous'] = previous_chunk is not None
        result['has_next'] = next_chunk is not None
        
        return result
    
    def approach_1_single_chunk_optimized(self, query: str, model: str, best_chunk: Dict) -> Dict:
        """Optimized Approach 1: Use only the most similar chunk (pre-calculated)"""
        print(f"\n=== Approach 1: Single Chunk Retrieval ===")
        print(f"Using pre-found chunk: {best_chunk['chunk_id']}")
        
        # Create prompt
        prompt = f"""Based on the following document excerpt, please answer the question.

Document excerpt:
{best_chunk['content']}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "single_chunk"
        result['chunks_used'] = 1
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = best_chunk['page_range']
        result['chunk_ids'] = [best_chunk['chunk_id']]
        
        return result
    
    def approach_2_sequential_chunks_optimized(self, query: str, model: str, best_chunk: Dict) -> Dict:
        """Optimized Approach 2: Use the most similar chunk + next chunk (pre-calculated)"""
        print(f"\n=== Approach 2: Sequential Chunks Retrieval ===")
        print(f"Using pre-found chunk: {best_chunk['chunk_id']}")
        
        # Get next chunk
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = [best_chunk]
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found - using only the best chunk")
        
        # Combine content
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpt{"s" if len(chunks_to_use) > 1 else ""}, please answer the question.

Document excerpt{"s" if len(chunks_to_use) > 1 else ""}:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "sequential_chunks"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        result['has_next'] = next_chunk is not None
        
        return result
    
    def approach_4_context_window_optimized(self, query: str, model: str, best_chunk: Dict) -> Dict:
        """Optimized Approach 3: Use previous + current + next chunk (pre-calculated)"""
        print(f"\n=== Approach 3: Context Window Retrieval (Prev + Current + Next) ===")
        print(f"Using pre-found chunk: {best_chunk['chunk_id']}")
        
        # Get previous and next chunks
        previous_chunk = self.get_previous_chunk(best_chunk['chunk_id'])
        next_chunk = self.get_next_chunk(best_chunk['chunk_id'])
        
        chunks_to_use = []
        
        # Add previous chunk if available
        if previous_chunk:
            chunks_to_use.append(previous_chunk)
            print(f"Previous chunk: {previous_chunk['chunk_id']}")
        else:
            print("No previous chunk found")
        
        # Add the best matching chunk
        chunks_to_use.append(best_chunk)
        
        # Add next chunk if available
        if next_chunk:
            chunks_to_use.append(next_chunk)
            print(f"Next chunk: {next_chunk['chunk_id']}")
        else:
            print("No next chunk found")
        
        print(f"Using {len(chunks_to_use)} chunks in context window")
        
        # Combine content in order
        combined_content = "\n\n".join([chunk['content'] for chunk in chunks_to_use])
        
        # Create prompt
        prompt = f"""Based on the following document excerpt{"s" if len(chunks_to_use) > 1 else ""} (in sequential order), please answer the question.

Document excerpt{"s" if len(chunks_to_use) > 1 else ""}:
{combined_content}

Question: {query}

Answer:"""
        
        # Query Ollama
        result = self.query_ollama(prompt, model)
        result['approach'] = "context_window"
        result['chunks_used'] = len(chunks_to_use)
        result['similarity_score'] = best_chunk['similarity']
        result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
        result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
        result['has_previous'] = previous_chunk is not None
        result['has_next'] = next_chunk is not None
        
        return result
    
    def approach_5_top_k_reranked(self, query: str, model: str, top_chunks: List[Dict]) -> Dict:
        """
        NEW Approach 5: Use top-5 reranked chunks (RECOMMENDED)
        This addresses the core limitation of betting on a single chunk
        """
        print(f"\n=== Approach 5: Top-K Reranked Chunks (RECOMMENDED) ===")
        
        result = {
            'approach': 'top_k_reranked',
            'success': False,
            'chunks_used': len(top_chunks),
            'response': '',
            'response_time': 0,
            'error': None
        }
        
        try:
            # Combine top-k chunks with clear separation
            combined_content = ""
            total_chars = 0
            chunks_to_use = []
            
            # Limit context size to avoid Ollama 500 errors (max ~8000 chars)
            MAX_CONTEXT_CHARS = 8000
            
            for i, chunk in enumerate(top_chunks, 1):
                chunk_text = chunk['content']
                chunk_header = f"\n--- Context {i} (Pages {chunk['page_range']}, Relevance: {chunk['rerank_score']:.3f}) ---\n"
                
                # Check if adding this chunk would exceed limit
                if total_chars + len(chunk_header) + len(chunk_text) > MAX_CONTEXT_CHARS:
                    print(f"  ⚠ Context size limit reached, using {len(chunks_to_use)} chunks instead of {len(top_chunks)}")
                    break
                
                combined_content += chunk_header + chunk_text + "\n"
                total_chars += len(chunk_header) + len(chunk_text)
                chunks_to_use.append(chunk)
            
            # Update chunks_used to actual number
            result['chunks_used'] = len(chunks_to_use)
            
            # Improved prompt with strict instructions
            prompt = f"""You are answering questions based ONLY on the provided context below.

STRICT RULES:
1. If the answer is not explicitly stated in the context, respond: "Not found in context"
2. Do NOT infer, guess, or use external knowledge
3. If the question requires numbers, include exact values from the context
4. If the question asks for a list, provide all items mentioned in the context
5. Cite which context section(s) you used (e.g., "According to Context 1...")

CONTEXT:
{combined_content}

QUESTION: {query}

ANSWER (following the strict rules above):"""
            
            print(f"  Using {len(chunks_to_use)} chunks ({total_chars} chars):")
            for i, chunk in enumerate(chunks_to_use, 1):
                print(f"    {i}. {chunk['chunk_id']} (Pages {chunk['page_range']}, Rerank: {chunk['rerank_score']:.3f})")
            
            # Query Ollama with timeout
            start_time = time.time()
            response = self.query_ollama(prompt, model, max_tokens=1000)
            end_time = time.time()
            
            if response['success']:
                result['success'] = True
                result['response'] = response['response']
                result['response_time'] = round(end_time - start_time, 2)
                result['similarity_score'] = chunks_to_use[0]['similarity']
                result['rerank_scores'] = [f"{c['rerank_score']:.3f}" for c in chunks_to_use]
                result['source_pages'] = ", ".join([chunk['page_range'] for chunk in chunks_to_use])
                result['chunk_ids'] = [chunk['chunk_id'] for chunk in chunks_to_use]
                result['context_chars'] = total_chars
                
                print(f"  ✓ Response generated in {result['response_time']}s")
            else:
                result['error'] = response.get('error', 'Unknown error')
                print(f"  ✗ Failed: {result['error']}")
        
        except Exception as e:
            result['error'] = str(e)
            print(f"  ✗ Error: {e}")
        
        return result
    
    def compare_approaches(self, query: str, model: str = "mistral:latest") -> Dict:
        """Compare all approaches with improved retrieval (top-k=15, reranking, top-5 chunks)"""
        print(f"\n{'='*80}")
        print(f"COMPARING GRAPHRAG APPROACHES (IMPROVED RETRIEVAL)")
        print(f"Query: {query}")
        print(f"Model: {model}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve top-15 candidates (better recall)
        print("\n[Step 1] Retrieving top-15 candidate chunks...")
        query_embedding = self.embed_query(query)
        similar_chunks = self.find_similar_chunks(query_embedding, top_k=15)
        
        if not similar_chunks:
            return {"error": "No chunks found for any approach", "success": False}
        
        print(f"  Retrieved {len(similar_chunks)} candidates")
        print(f"  Similarity range: {similar_chunks[0]['similarity']:.3f} to {similar_chunks[-1]['similarity']:.3f}")
        
        # Step 2: Rerank based on query type
        print("\n[Step 2] Reranking chunks based on query characteristics...")
        reranked_chunks = self.rerank_chunks(similar_chunks, query)
        
        print(f"  Top 3 after reranking:")
        for i, chunk in enumerate(reranked_chunks[:3], 1):
            print(f"    {i}. {chunk['chunk_id']} - Similarity: {chunk['similarity']:.3f}, Rerank: {chunk['rerank_score']:.3f}")
        
        # Step 3: Use top-5 chunks for context (not just 1)
        best_chunk = reranked_chunks[0]
        
        results = {}
        
        # Approach 1: Single best chunk (baseline)
        results['approach_1'] = self.approach_1_single_chunk_optimized(query, model, best_chunk)
        
        # Approach 2: Sequential chunks (best + next)
        results['approach_2'] = self.approach_2_sequential_chunks_optimized(query, model, best_chunk)
        
        # Approach 3: Context window (prev + best + next)
        results['approach_3'] = self.approach_4_context_window_optimized(query, model, best_chunk)
        
        # NEW Approach 4: Top-5 reranked chunks (RECOMMENDED)
        results['approach_4'] = self.approach_5_top_k_reranked(query, model, reranked_chunks[:5])
        
        # Summary
        print(f"\n{'='*80}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for approach_name, result in results.items():
            if result['success']:
                approach_label = {
                    'approach_1': 'Single Chunk (Baseline)',
                    'approach_2': 'Sequential Chunks',
                    'approach_3': 'Context Window',
                    'approach_4': 'Top-5 Reranked (RECOMMENDED)'
                }.get(approach_name, approach_name)
                
                print(f"\n{approach_label}:")
                print(f"  Chunks used: {result['chunks_used']}")
                print(f"  Response time: {result['response_time']}s")
                print(f"  Similarity score: {result.get('similarity_score', 'N/A'):.3f}")
                print(f"  Source pages: {result['source_pages']}")
                if 'rerank_scores' in result:
                    print(f"  Rerank scores: {result['rerank_scores']}")
                print(f"  Response preview: {result['response'][:200]}...")
            else:
                print(f"\n{approach_name.upper().replace('_', ' ')}: FAILED")
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        return results
    
    def close(self):
        """Close connections"""
        if self.neo4j_driver:
            self.neo4j_driver.close()

def main():
    """Main function to test GraphRAG system"""
    
    # Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Lexical12345")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
    
    print("GraphRAG System with Neo4j Knowledge Graph and Ollama")
    print("=" * 60)
    
    try:
        # Initialize system
        graphrag = GraphRAGSystem(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OLLAMA_URL)
        
        # Get available models
        models = graphrag.get_available_ollama_models()
        if not models:
            print("No Ollama models found. Please install a model first:")
            print(f"  ollama pull {OLLAMA_MODEL}")
            return
        
        # Use specified model or first available
        model_to_use = OLLAMA_MODEL if OLLAMA_MODEL in models else models[0]
        print(f"Using model: {model_to_use}")
        
        if model_to_use != OLLAMA_MODEL:
            print(f"Note: Requested model '{OLLAMA_MODEL}' not found, using '{model_to_use}' instead")
        
        # Test queries
        test_queries = [
            "What was Aramco's net income for the full year 2024 in Saudi Riyals?",
            "What was Aramco's net income for the full year 2023 in Saudi Riyals?",
            "By what percentage did Aramco's net income decrease from 2023 to 2024?",
            "What was Aramco's gearing ratio as of December 31, 2024?",
        ]
        
        # Test each query with all approaches
        all_results = {}
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n\n{'#'*100}")
            print(f"TEST QUERY {i}: {query}")
            print(f"{'#'*100}")
            
            results = graphrag.compare_approaches(query, model_to_use)
            all_results[f"query_{i}"] = {
                "query": query,
                "results": results
            }
        
        # Save results to file
        output_file = "/app/output/graphrag_comparison_results.json" if os.path.exists("/app/output") else "graphrag_comparison_results.json"
        
        # Add metadata to results
        final_results = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": model_to_use,
                "neo4j_uri": NEO4J_URI,
                "ollama_url": OLLAMA_URL,
                "total_queries": len(test_queries),
                "approaches": [
                    {"id": "approach_1", "name": "Single Chunk", "description": "Uses only the most similar chunk (baseline)"},
                    {"id": "approach_2", "name": "Sequential Chunks", "description": "Uses the most similar chunk + next chunk"},
                    {"id": "approach_3", "name": "Context Window", "description": "Uses previous + current + next chunk"},
                    {"id": "approach_4", "name": "Top-5 Reranked", "description": "Uses top-5 reranked chunks (RECOMMENDED)"}
                ]
            },
            "queries": all_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n\nResults saved to: {output_file}")
        
        # Performance summary
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        approach_stats = {
            'approach_1': {'total_time': 0, 'success_count': 0},
            'approach_2': {'total_time': 0, 'success_count': 0},
            'approach_3': {'total_time': 0, 'success_count': 0},
            'approach_4': {'total_time': 0, 'success_count': 0}
        }
        
        for query_data in all_results.values():
            for approach, result in query_data['results'].items():
                if result['success']:
                    approach_stats[approach]['total_time'] += result['response_time']
                    approach_stats[approach]['success_count'] += 1
        
        for approach, stats in approach_stats.items():
            if stats['success_count'] > 0:
                avg_time = stats['total_time'] / stats['success_count']
                approach_names = {
                    'approach_1': 'Single Chunk (Baseline)',
                    'approach_2': 'Sequential Chunks',
                    'approach_3': 'Context Window',
                    'approach_4': 'Top-5 Reranked (RECOMMENDED)'
                }
                approach_name = approach_names.get(approach, approach.replace('_', ' ').title())
                print(f"{approach_name}: {avg_time:.2f}s average, {stats['success_count']}/{len(test_queries)} successful")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'graphrag' in locals():
            graphrag.close()

if __name__ == "__main__":
    main()