"""
Query complexity classifier to determine optimal processing strategy.
"""

from typing import Dict, Literal
from src.rag.query.llm_client import LLMClient


class QueryComplexityClassifier:
    """Classifies query complexity to optimize token usage."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def classify(self, query: str) -> Dict[str, any]:
        """
        Classify query complexity and recommend processing strategy.
        
        Args:
            query: User's question
            
        Returns:
            Dict with classification and recommendations
        """
        # Simple heuristic-based classification (fast, no tokens)
        complexity = self._heuristic_classify(query)
        
        # Only use LLM for borderline cases
        if complexity['confidence'] < 0.8:
            complexity = self._llm_classify(query)
        
        return complexity
    
    def _heuristic_classify(self, query: str) -> Dict:
        """Fast heuristic classification without token usage."""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Simple query indicators
        simple_patterns = [
            'what is', 'define', 'explain', 'how does', 'who is',
            'when was', 'where is'
        ]
        
        # Complex query indicators
        complex_patterns = [
            'compare', 'contrast', 'difference between', 'similarities',
            'analyze', 'evaluate', 'discuss', 'multiple', 'various',
            'across papers', 'in different', 'how do they differ'
        ]
        
        # Multi-part indicators
        multipart_indicators = [' and ', ' or ', '?', ';', ',']
        question_marks = query.count('?')
        
        # Calculate complexity score
        complexity_score = 0
        
        # Word count factor
        if word_count > 20:
            complexity_score += 0.3
        elif word_count > 10:
            complexity_score += 0.1
        
        # Pattern matching
        if any(pattern in query_lower for pattern in complex_patterns):
            complexity_score += 0.4
        elif any(pattern in query_lower for pattern in simple_patterns):
            complexity_score -= 0.2
        
        # Multi-part question
        if question_marks > 1 or sum(1 for ind in multipart_indicators if ind in query) > 2:
            complexity_score += 0.3
        
        # Classify
        if complexity_score < 0.3:
            level = 'simple'
            use_agent = False
            use_validation = False
            confidence = 0.85
        elif complexity_score < 0.6:
            level = 'moderate'
            use_agent = False
            use_validation = True
            confidence = 0.75
        else:
            level = 'complex'
            use_agent = True
            use_validation = True
            confidence = 0.85
        
        return {
            'level': level,
            'score': complexity_score,
            'use_agent': use_agent,
            'use_validation': use_validation,
            'use_expansion': True,  # Always cheap and effective
            'confidence': confidence,
            'reasoning': f"Word count: {word_count}, complexity indicators detected"
        }
    
    def _llm_classify(self, query: str) -> Dict:
        """LLM-based classification for uncertain cases (~100 tokens)."""
        prompt = f"""Classify this query's complexity for a RAG system.

Query: {query}

Classify as:
- SIMPLE: Direct factual question, single concept
- MODERATE: Multiple concepts, some reasoning needed
- COMPLEX: Multi-hop reasoning, comparisons, or multiple questions

Format:
COMPLEXITY: <SIMPLE/MODERATE/COMPLEX>
REASONING: <brief explanation>"""
        
        response = self.llm_client.generate(prompt)
        
        # Parse response
        if 'COMPLEXITY:' in response:
            complexity_line = response.split('COMPLEXITY:')[1].split('\n')[0].strip().lower()
            
            if 'simple' in complexity_line:
                level = 'simple'
                use_agent = False
                use_validation = False
            elif 'complex' in complexity_line:
                level = 'complex'
                use_agent = True
                use_validation = True
            else:
                level = 'moderate'
                use_agent = False
                use_validation = True
        else:
            # Fallback to moderate
            level = 'moderate'
            use_agent = False
            use_validation = True
        
        reasoning = response.split('REASONING:')[1].strip() if 'REASONING:' in response else "LLM classification"
        
        return {
            'level': level,
            'use_agent': use_agent,
            'use_validation': use_validation,
            'use_expansion': True,
            'confidence': 0.9,
            'reasoning': reasoning
        }
