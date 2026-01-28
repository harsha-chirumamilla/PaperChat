"""
ReAct (Reasoning + Acting) agent for complex RAG queries.
Implements iterative reasoning with tool use for multi-hop questions.
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from src.rag.query.llm_client import LLMClient
from src.common.config import Config
import logging

logger = logging.getLogger(__name__)


class AgentAction(Enum):
    """Available actions for the ReAct agent."""
    SEARCH = "search"
    DECOMPOSE = "decompose"
    SYNTHESIZE = "synthesize"
    FINISH = "finish"


class ReActAgent:
    """
    Reasoning + Acting agent for complex RAG queries.
    
    Implements ReAct pattern:
    1. Thought: Analyze the query
    2. Action: Choose tool/approach
    3. Observation: Process results
    4. Repeat until answer is found
    """
    
    def __init__(self, llm_client: LLMClient, rag_system):
        """
        Initialize ReAct agent.
        
        Args:
            llm_client: LLM client for reasoning
            rag_system: RAGSystem instance for document retrieval
        """
        self.llm = llm_client
        self.rag = rag_system
        self.max_iterations = Config.AGENT_MAX_ITERATIONS
        logger.info("ReAct agent initialized with max_iterations=%d", self.max_iterations)
    
    def run(self, query: str) -> Dict:
        """
        Execute ReAct loop for complex query.
        
        Args:
            query: User's question
            
        Returns:
            Dict with answer, reasoning trace, and sources
        """
        logger.info("Starting ReAct agent for query: %s", query)
        reasoning_trace = []
        context_docs = []
        
        for iteration in range(self.max_iterations):
            logger.info("Iteration %d/%d", iteration + 1, self.max_iterations)
            
            # THOUGHT: Analyze current state
            thought = self._generate_thought(query, reasoning_trace)
            reasoning_trace.append({"type": "thought", "content": thought})
            logger.debug("Thought: %s", thought)
            
            # ACTION: Decide what to do
            action, action_input = self._decide_action(thought, query)
            reasoning_trace.append({
                "type": "action",
                "action": action.value,
                "input": action_input
            })
            logger.info("Action: %s with input: %s", action.value, action_input[:50])
            
            # OBSERVATION: Execute action
            if action == AgentAction.DECOMPOSE:
                sub_queries = self._decompose_query(query)
                observation = "Decomposed into: " + "; ".join(sub_queries)
                
                # Gather context for each sub-query
                for sq in sub_queries:
                    docs = self.rag.retriever.retrieve(sq)
                    context_docs.extend(docs)
                logger.info("Retrieved %d documents from %d sub-queries", len(context_docs), len(sub_queries))
                
            elif action == AgentAction.SEARCH:
                docs = self.rag.retriever.retrieve(action_input)
                context_docs.extend(docs)
                observation = "Retrieved %d documents" % len(docs)
                logger.info("Retrieved %d documents for search", len(docs))
                
            elif action == AgentAction.SYNTHESIZE:
                # Rerank all gathered documents
                if context_docs:
                    reranked = self.rag.reranker.rerank(query, context_docs)
                    # Generate final answer
                    prompt = self.rag.prompter.format_rag_prompt(query, reranked)
                    answer = self.llm.generate(prompt)
                    
                    logger.info("Synthesized answer from %d documents", len(reranked))
                    return {
                        "answer": answer,
                        "reasoning_trace": reasoning_trace,
                        "sources": reranked,
                        "iterations": iteration + 1
                    }
                else:
                    observation = "No documents found to synthesize"
                    logger.warning("No documents to synthesize")
                    
            elif action == AgentAction.FINISH:
                # Direct answer without further retrieval
                if context_docs:
                    reranked = self.rag.reranker.rerank(query, context_docs)
                    prompt = self.rag.prompter.format_rag_prompt(query, reranked)
                else:
                    prompt = self.rag.prompter.format_rag_prompt(query, [])
                
                answer = self.llm.generate(prompt)
                
                logger.info("Finished after %d iterations", iteration + 1)
                return {
                    "answer": answer,
                    "reasoning_trace": reasoning_trace,
                    "sources": context_docs,
                    "iterations": iteration + 1
                }
            
            reasoning_trace.append({"type": "observation", "content": observation})
            logger.debug("Observation: %s", observation)
        
        # Max iterations reached
        logger.warning("Max iterations reached without completion")
        return {
            "answer": "Could not find sufficient information after multiple attempts.",
            "reasoning_trace": reasoning_trace,
            "sources": context_docs,
            "iterations": self.max_iterations
        }
    
    def _generate_thought(self, query: str, trace: List[Dict]) -> str:
        """Generate reasoning thought based on current state."""
        prompt = """You are analyzing a research question to determine the best approach.

            Question: %s

            Previous steps: %d

            Analyze the question and determine:
            1. Is this a simple factual question or complex multi-part question?
            2. Do we need to search for more information?
            3. Can we synthesize an answer now?

            Provide your analysis in 2-3 sentences.""" % (query, len(trace) // 3)
        
        return self.llm.generate(prompt)
    
    def _decide_action(self, thought: str, query: str) -> Tuple[AgentAction, str]:
        """Decide next action based on thought."""
        prompt = """Based on this analysis, choose the next action:

            Analysis: %s
            Question: %s

            Available actions:
            - decompose: Break complex question into sub-questions
            - search: Search for specific information
            - synthesize: Combine gathered information into answer
            - finish: Provide final answer

            Choose ONE action and provide input. Format:
            ACTION: <action_name>
            INPUT: <specific input for action>""" % (thought, query)
        
        response = self.llm.generate(prompt)
        
        # Parse response
        lines = response.strip().split('\n')
        action_line = next((l for l in lines if l.startswith('ACTION:')), 'ACTION: search')
        input_line = next((l for l in lines if l.startswith('INPUT:')), 'INPUT: ' + query)
        
        action_str = action_line.split(':', 1)[1].strip().lower()
        action_input = input_line.split(':', 1)[1].strip()
        
        # Map to enum
        action_map = {
            'decompose': AgentAction.DECOMPOSE,
            'search': AgentAction.SEARCH,
            'synthesize': AgentAction.SYNTHESIZE,
            'finish': AgentAction.FINISH
        }
        
        action = action_map.get(action_str, AgentAction.SEARCH)
        return action, action_input
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-questions."""
        prompt = """Break this complex question into 2-3 simpler sub-questions:

            Question: %s

            Sub-questions (one per line):""" % query
                    
        response = self.llm.generate(prompt)
        sub_queries = [
            line.strip().lstrip('123456789.- ')
            for line in response.split('\n')
            if line.strip() and len(line.strip()) > 10
        ]
        
        return sub_queries[:3]  # Max 3 sub-queries
