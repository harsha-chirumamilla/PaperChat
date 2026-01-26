from typing import Dict
from src.rag.query.llm_client import LLMClient


class AnswerValidator:
    """Validates generated answers for accuracy and groundedness."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def validate_answer(
        self,
        query: str,
        answer: str,
        context: str,
        threshold: float = 0.7
    ) -> Dict:
        """
        Check if answer is grounded in provided context.
        
        Args:
            query: Original question
            answer: Generated answer
            context: Retrieved context
            threshold: Confidence threshold
            
        Returns:
            Dict with validation results
        """
        prompt = f"""Evaluate if the answer is fully grounded in the provided context.

Question: {query}

Context: {context}

Answer: {answer}

Analysis:
1. Are all factual claims in the answer supported by the context? (Yes/No)
2. Does the answer introduce information not in the context? (Yes/No)
3. Is the answer directly addressing the question? (Yes/No)
4. Confidence score (0.0-1.0):

Provide your evaluation in this format:
GROUNDED: <Yes/No>
HALLUCINATION: <Yes/No>
RELEVANT: <Yes/No>
CONFIDENCE: <0.0-1.0>
ISSUES: <list any problems or "None">"""
        
        validation = self.llm.generate(prompt)
        
        # Parse validation
        result = {
            "grounded": "Yes" in validation.split("GROUNDED:")[1].split("\n")[0] if "GROUNDED:" in validation else False,
            "hallucination": "Yes" in validation.split("HALLUCINATION:")[1].split("\n")[0] if "HALLUCINATION:" in validation else False,
            "relevant": "Yes" in validation.split("RELEVANT:")[1].split("\n")[0] if "RELEVANT:" in validation else True,
            "confidence": 0.5,  # Default
            "issues": []
        }
        
        # Extract confidence
        try:
            if "CONFIDENCE:" in validation:
                conf_line = validation.split("CONFIDENCE:")[1].split("\n")[0]
                result["confidence"] = float(conf_line.strip())
        except:
            pass
        
        # Extract issues
        if "ISSUES:" in validation:
            issues_text = validation.split("ISSUES:")[1].strip()
            if issues_text.lower() != "none":
                result["issues"] = [issues_text]
        
        result["valid"] = (
            result["grounded"] and 
            not result["hallucination"] and
            result["relevant"] and
            result["confidence"] >= threshold
        )
        
        return result
    
    def improve_answer(
        self,
        query: str,
        answer: str,
        context: str,
        validation: Dict
    ) -> str:
        """
        Improve answer based on validation feedback.
        
        Args:
            query: Original question
            answer: Initial answer
            context: Retrieved context
            validation: Validation results
            
        Returns:
            Improved answer
        """
        if validation["valid"]:
            return answer
        
        issues_text = "; ".join(validation["issues"]) if validation["issues"] else "answer quality"
        
        prompt = f"""The previous answer has issues: {issues_text}

Please generate an improved answer that:
1. Only uses information from the provided context
2. Directly addresses the question
3. Cites specific sources when making claims

Question: {query}

Context: {context}

Previous answer: {answer}

Improved answer:"""
        
        return self.llm.generate(prompt)
