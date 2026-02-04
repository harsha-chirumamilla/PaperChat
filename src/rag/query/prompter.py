from typing import List
from langchain_core.documents import Document


class PromptFormatter:
    """Formats prompts for LLM generation with retrieved context."""
    
    @staticmethod
    def format_rag_prompt(query: str, context_docs: List[Document]) -> str:
        """
        Formats the prompt with retrieved context and numbered citations.
        
        Args:
            query: User's question
            context_docs: Retrieved and reranked documents
            
        Returns:
            Formatted prompt string
        """
        context_text = "\n\n".join(
            [
                f"[{i+1}] {doc.page_content} (arXiv:{doc.metadata.get('source', 'N/A')})"
                for i, doc in enumerate(context_docs)
            ]
        )
        
        return f"""Answer the question using only the context below. When referencing information from the context, cite the source using numbered citations [1], [2], etc. that correspond to the context sources. Use inline citations in the text where you reference specific information.

Context:
{context_text}

Question: {query}

Provide your answer with inline citations in the format [1], [2], etc."""
