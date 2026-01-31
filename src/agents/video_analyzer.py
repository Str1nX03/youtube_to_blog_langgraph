import json
import yt_dlp
import requests
import sys
import os
from typing import TypedDict, Optional, Dict, Any
from src.logger import logging
from src.exception import CustomException
from src.utils import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

class AgentState(TypedDict):
    video_url: str
    transcript: Optional[str]
    analysis: Optional[str]
    error: Optional[str]

class YoutubeAnalyzeAgent:

    def __init__(self, llm=None):
        # Use provided LLM or fetch default if None
        self.llm = llm if llm else get_llm()
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            graph = StateGraph(AgentState)

            # 1. Add Processing Nodes
            graph.add_node("fetch_transcript", self._fetch_transcript)
            graph.add_node("analyze_transcript", self._analyze_transcript)
            
            # NOTE: "check_extraction" is NOT added as a node because it is a routing function.

            # 2. Add Entry Point
            graph.add_edge(START, "fetch_transcript")

            # 3. Add Conditional Edge (Router)
            # FIXED: Passed self._check_extraction (the function) instead of "check_extraction" (string)
            graph.add_conditional_edges(
                "fetch_transcript",
                self._check_extraction,
                {
                    "analyze": "analyze_transcript",
                    "end": END
                }
            )

            # 4. Add Final Edge
            graph.add_edge("analyze_transcript", END)

            return graph.compile()
        
        except Exception as e:
            raise CustomException(e, sys)

    def _fetch_transcript(self, state: AgentState):
        try:
            video_url = state["video_url"]
            is_vercel = os.environ.get('VERCEL') or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
            cookies_arg = None

            env_cookies = os.environ.get("YOUTUBE_COOKIES")
            if env_cookies:
                try:
                    temp_cookies_path = "/tmp/cookies.txt"
                    with open(temp_cookies_path, "w") as f:
                        f.write(env_cookies)
                    cookies_arg = temp_cookies_path
                except Exception as e:
                    raise CustomException(e, sys)
            
            elif os.path.exists("cookies.txt"):
                cookies_arg = "cookies.txt"

            ydl_opts = {
                'skip_download': True,
                'format': 'best',
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'hi', 'ja', 'es'],
                'cookiefile': cookies_arg,
                'quiet': True,
                'no_warnings': True,
                'cache_dir': '/tmp/yt-dlp-cache' if is_vercel else None,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_call_home': True,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    try:
                        info = ydl.extract_info(video_url, download=False)
                    except Exception as e:
                        raise CustomException(e, sys)

                    if not info:
                        return {"error": "yt-dlp returned no information."}
                    
                    subtitles = info.get("subtitles", {})
                    auto_captions = info.get("automatic_captions", {})
                    all_subs = {**auto_captions, **subtitles}

                    if not all_subs:
                        return {"error": "No subtitles found in video metadata."}
                    
                    chosen_lang = None
                    for lang in ["en", "hi", "ja", "es"]:
                        if lang in all_subs:
                            chosen_lang = lang
                            break
                    
                    if not chosen_lang:
                        chosen_lang = list(all_subs.keys())[0]

                    subs_list = all_subs.get(chosen_lang, [])
                    json3_url = next((sub.get('url') for sub in subs_list if sub.get('ext') == 'json3'), None)

                    if not json3_url:
                        json3_url = subs_list[0].get("url")

                    if not json3_url:
                        return {"error": "Could not find a valid subtitle URL."}
                    
                    session = requests.Session()
                    session.headers.update({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    })
                    response = session.get(json3_url)

                    final_transcript = ""
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            events = data.get("events", [])
                            full_text = []
                            for event in events:
                                segs = event.get("segs", [])
                                for seg in segs:
                                    txt = seg.get("utf8", "").strip()
                                    if txt:
                                        full_text.append(txt)
                            final_transcript = " ".join(full_text)
                        except Exception as e:
                            raise CustomException(e, sys)
                    else:
                        return {"error": f"Failed to download subs. Status: {response.status_code}"}

                    return {"transcript": final_transcript}
                
            except Exception as e:
                raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e, sys)

    def _analyze_transcript(self, state: AgentState):
        try:
            transcript = state.get("transcript")

            if not transcript:
                return {"error": "No transcript available for analysis."}
            
            truncated_transcript = transcript[:15000]
            system_prompt = (
                "You are an expert video content analyst. "
                "Your goal is to extract the core topics, key takeaways, and tone from video transcripts."
            )

            user_prompt = f"""
                Analyze the following YouTube Video Transcript.
                
                NOTE: The transcript might be in a foreign language. 
                You MUST translate the concepts and Output the final analysis in ENGLISH.

                Transcript (Truncated):
                {truncated_transcript} 

                Output a structured summary containing:
                1. Main Topic
                2. Key Points (Bullet points)
                3. The tone of the video
                4. Important keywords
            """

            try:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]

                response = self.llm.invoke(messages)
                return {"analysis": response.content} # Ensure we return content, not just the object if using certain LLMs
            
            except Exception as e:
                raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e, sys)
    
    def _check_extraction(self, state: AgentState):
        try:
            # Check for errors or missing transcript to decide path
            if state.get("error"):
                return "end"
            
            if not state.get("transcript"):
                return "end"
            
            return "analyze"

        except Exception as e:
            raise CustomException(e, sys)

    def run(self, video_url: str):
        try:
            initial_state = {
                "video_url": video_url,
                "transcript": None,
                "analysis": None,
                "error": None
            }
            final_state = self.graph.invoke(initial_state)
            return final_state
        
        except Exception as e:
            raise CustomException(e, sys)