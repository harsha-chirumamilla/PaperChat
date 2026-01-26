from typing import List, Dict
from langchain_core.documents import Document
from src.rag.query.llm_client import LLMClient
from src.common.config import Config


class QueryExpander:
    """Expands user queries into multiple perspectives for better retrieval."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def expand_query(self, query: str, num_variants: int = 3) -> List[str]:
        """
        Generate multiple query variations using LLM.
        
        Args:
            query: Original user query
            num_variants: Number of query variants to generate
            
        Returns:
            List of query variants including original
        """
        prompt = f"""Generate {num_variants} alternative phrasings of this question for academic paper search. 
Keep each variant focused and concise (max 15 words).

Original: {query}

Variants (one per line):"""
        
        response = self.llm_client.generate(prompt)
        variants = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Always include original query
        return [query] + variants[:num_variants]
    
    def retrieve_with_fusion(self, queries: List[str], retriever) -> List[Document]:
        """
        Retrieve documents for all queries and apply Reciprocal Rank Fusion.
        
        Args:
            queries: List of query variants
            retriever: Hybrid retriever instance
            
        Returns:
            Fused and ranked documents
        """
        # Retrieve for each query
        all_results = {}
        
        for rank_offset, query_variant in enumerate(queries):
            docs = retriever.retrieve(query_variant)
            
            for rank, doc in enumerate(docs):
                key = doc.page_content
                if key not in all_results:
                    all_results[key] = {"doc": doc, "score": 0.0}
                
                # Reciprocal Rank Fusion formula
                # Give slight preference to original query (rank_offset=0)
                weight = 1.0 if rank_offset == 0 else 0.8
                all_results[key]["score"] += weight / (rank + 60)
        
        # Sort by fused score
        ranked = sorted(
            all_results.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        
        return [item["doc"] for item in ranked[:Config.RETRIEVAL_K]]
