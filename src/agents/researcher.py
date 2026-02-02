import sys
import json
import logging
from typing import TypedDict, Optional, List, Any

# Standard import for the tool
try:
    from langchain_community.tools import DuckDuckGoSearchRun
except ImportError:
    # Fallback for some versions
    from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from src.exception import CustomException
from src.utils import get_llm

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- STATE DEFINITION ---
class ResearchState(TypedDict):
    video_analysis: str
    search_queries: Optional[List[str]]
    research_summary: Optional[str]
    error: Optional[str]

class ResearchAgent:
    def __init__(self, llm=None):
        # Use provided LLM or fetch default if None
        self.llm = llm if llm else get_llm()
        # Initialize the LangChain DuckDuckGo tool
        self.search_tool = DuckDuckGoSearchRun()
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            graph = StateGraph(ResearchState)

            # 1. Add Processing Nodes
            graph.add_node("generate_queries", self._generate_queries)
            graph.add_node("perform_research", self._perform_research)

            # 2. Add Entry Point
            graph.add_edge(START, "generate_queries")

            # 3. Add Conditional Edge (Router)
            # Passing the function self._check_queries which returns "research" or "end"
            graph.add_conditional_edges(
                "generate_queries",
                self._check_queries,
                {
                    "research": "perform_research",
                    "end": END
                }
            )

            # 4. Add Final Edge
            graph.add_edge("perform_research", END)

            return graph.compile()

        except Exception as e:
            raise CustomException(e, sys)

    def _search_web(self, query: str) -> str:
        """Helper method to perform the search using LangChain's tool."""
        try:
            logging.info(f"Searching web for: {query}")
            # Invoke the tool
            result = self.search_tool.invoke(query)
            return result
        except Exception as e:
            logging.error(f"Search failed for query '{query}': {str(e)}")
            return f"Error searching for {query}: {str(e)}\n"

    def _generate_queries(self, state: ResearchState):
        """Node 1: LLM generates search queries based on the video analysis."""
        try:
            video_analysis = state.get("video_analysis")
            
            if not video_analysis:
                return {"error": "No video analysis provided for research context."}

            logging.info("Generating search queries based on video analysis...")

            search_plan_prompt = f"""
                Based on the following video analysis, generate 3 specific, high-quality search queries to find the latest updates, confirmed news, or verified facts.
                
                Video Analysis:
                {video_analysis}
                
                OUTPUT FORMAT:
                Return ONLY a raw JSON list of strings. Do not use Markdown code blocks.
                Example: ["query 1", "query 2", "query 3"]
            """

            messages = [
                SystemMessage(content="You are a senior web researcher. Generate precise search queries."),
                HumanMessage(content=search_plan_prompt)
            ]

            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            # Robust Parsing Logic
            queries = []
            try:
                # Attempt to clean markdown if present
                clean_raw = content.replace('```json', '').replace('```', '').strip()
                queries = json.loads(clean_raw)
            except json.JSONDecodeError:
                logging.warning("JSON parsing failed, attempting manual line parsing.")
                # Fallback: Split by lines and clean up common list markers
                queries = [
                    line.strip().lstrip('1234567890. -"') 
                    for line in content.split('\n') 
                    if line.strip() and len(line) > 5
                ]

            if not queries or not isinstance(queries, list):
                return {"error": "Could not parse valid search queries from LLM response."}

            return {"search_queries": queries}

        except Exception as e:
            raise CustomException(e, sys)

    def _perform_research(self, state: ResearchState):
        """Node 2: Executes the search queries and aggregates results."""
        try:
            queries = state.get("search_queries", [])
            if not queries:
                return {"error": "No queries to search."}

            logging.info(f"Executing {len(queries)} search queries...")
            
            full_research_summary = "External Research Findings:\n\n"
            
            for query in queries:
                if isinstance(query, str) and len(query) > 2:
                    full_research_summary += f"--- Results for: {query} ---\n"
                    full_research_summary += self._search_web(query)
                    full_research_summary += "\n\n"

            logging.info("Web research complete.")
            return {"research_summary": full_research_summary}

        except Exception as e:
            raise CustomException(e, sys)

    def _check_queries(self, state: ResearchState):
        
        try:
            
            if state.get("error"):
                return "end"
            
            queries = state.get("search_queries")
            if not queries or not isinstance(queries, list) or len(queries) == 0:
                return "end"
            
            return "research"

        except Exception as e:
            logging.error(f"Error in router _check_queries: {e}")
            return "end"

    def run(self, video_analysis: str):
        """Entry point for the agent."""
        try:
            initial_state = {
                "video_analysis": video_analysis,
                "search_queries": [],
                "research_summary": None,
                "error": None
            }
            final_state = self.graph.invoke(initial_state)
            return final_state
        
        except Exception as e:
            raise CustomException(e, sys)