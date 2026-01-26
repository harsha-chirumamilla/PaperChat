import hashlib
import json
from typing import List, Optional
import redis
from langchain_core.documents import Document


class PromptCacheManager:
    """Manages prompt and context caching for API optimization."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
    
    def get_context_hash(self, documents: List[Document]) -> str:
        """Generate stable hash for document set."""
        content = json.dumps(
            [doc.page_content for doc in documents],
            sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def cache_context(self, context_hash: str, documents: List[Document]):
        """Cache document context."""
        key = f"context:{context_hash}"
        value = json.dumps([
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ])
        self.redis.setex(key, self.cache_ttl, value)
    
    def get_cached_context(self, context_hash: str) -> Optional[List[Document]]:
        """Retrieve cached context."""
        key = f"context:{context_hash}"
        cached = self.redis.get(key)
        
        if cached:
            data = json.loads(cached)
            return [
                Document(
                    page_content=item["content"],
                    metadata=item["metadata"]
                )
                for item in data
            ]
        return None
    
    def should_use_cache(self, query: str, context_hash: str) -> bool:
        """
        Determine if we should use cached context for this query.
        Only cache if similar query was asked recently.
        """
        # Store query patterns for context reuse
        pattern_key = f"pattern:{context_hash}"
        recent_queries = self.redis.lrange(pattern_key, 0, 9)  # Last 10 queries
        
        # Check semantic similarity (simple word overlap for now)
        query_words = set(query.lower().split())
        
        for recent_query in recent_queries:
            recent_words = set(recent_query.decode().lower().split())
            overlap = len(query_words & recent_words) / len(query_words | recent_words)
            
            if overlap > 0.6:  # 60% similarity threshold
                return True
        
        # Store this query pattern
        self.redis.lpush(pattern_key, query)
        self.redis.ltrim(pattern_key, 0, 9)
        self.redis.expire(pattern_key, self.cache_ttl)
        
        return False
