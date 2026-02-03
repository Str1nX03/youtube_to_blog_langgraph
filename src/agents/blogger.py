import sys
from typing import TypedDict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from src.exception import CustomException
from src.utils import get_llm

class AgentState(TypedDict):

    video_analysis: str
    research_findings: str
    blog_post: Optional[str]
    error: Optional[str]

class BloggerAgent:

    def __init__(self, llm = get_llm()):
        
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        try:

            graph = StateGraph(AgentState)

            graph.add_node("write_blog", self._write_blog)

            graph.add_conditional_edges(
                START,
                self._check_context,
                {
                    "generate": "write_blog",
                    "end": END
                }
            )
            graph.add_edge("write_blog", END)

            return graph.compile()

        except Exception as e:
            raise CustomException(e, sys)

    def _write_blog(self, state: AgentState):
        """Node: Generates the blog post using the LLM."""
        try:
            
            video_analysis = state.get("video_analysis")
            research_findings = state.get("research_findings")

            prompt = f"""
            Create a high-quality blog post based on the following information.
        
            SOURCE 1: Video Analysis (Core Content)
            {video_analysis}
            
            SOURCE 2: External Research (Latest Context)
            {research_findings}
            
            Requirements:
            - Catchy Title (Make it click-worthy)
            - Engaging Introduction (Hook the reader immediately)
            - Well-structured body with clear headers
            - Integrate the external research naturally to add value
            - Conclusion with a call to action
            - Use Markdown formatting
            - Tone: Fun, informative, and accessible to general readers
            """

            messages = [
                SystemMessage(content="You are a professional blog writer. You write engaging, viral-ready, and SEO-optimized articles."),
                HumanMessage(content=prompt)
            ]

            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)

            return {"blog_post": content}

        except Exception as e:
            raise CustomException(e, sys)

    def _check_context(self, state: AgentState):
        """
        Router: Checks if necessary inputs exist before attempting to write.
        CRITICAL: Must ALWAYS return 'generate' or 'end'.
        """
        try:
            
            if state.get("error"):
                return "end"
            v_analysis = state.get("video_analysis")
            r_findings = state.get("research_findings")

            if not v_analysis or not r_findings:
                return "end"
            
            return "generate"

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, video_analysis: str, research_findings: str):
        """Entry point for the agent."""
        try:
            initial_state = {
                "video_analysis": video_analysis,
                "research_findings": research_findings,
                "blog_post": None,
                "error": None
            }
            final_state = self.graph.invoke(initial_state)
            return final_state
        
        except Exception as e:
            raise CustomException(e, sys)