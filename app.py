from flask import Flask, render_template, request, jsonify
import sys
from src.agents.video_analyzer import YoutubeAnalyzeAgent
from src.agents.researcher import ResearchAgent
from src.agents.blogger import BloggerAgent
from src.exception import CustomException

app = Flask(__name__)

@app.route('/')
def index():
    """Renders the landing page."""
    return render_template('landing.html')

@app.route('/product')
def product():
    """Renders the product page."""
    return render_template('product.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """
    Orchestrates the AI agents:
    1. Analyzer: Video -> Transcript -> Analysis
    2. Researcher: Analysis -> Web Queries -> Research Summary
    3. Blogger: Analysis + Research -> Blog Post
    """
    data = request.json
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "No video URL provided"}), 400

    try:
        
        analyzer = YoutubeAnalyzeAgent()
        researcher = ResearchAgent()
        blogger = BloggerAgent()

        analyze_state = analyzer.run(video_url)
        
        if analyze_state.get("error"):
            return jsonify({"error": f"Analyzer Error: {analyze_state['error']}"}), 500
        
        video_analysis = analyze_state.get("analysis")
        if not video_analysis:
             return jsonify({"error": "Failed to generate video analysis"}), 500

        research_state = researcher.run(video_analysis)
        
        if research_state.get("error"):
            
            return jsonify({"error": f"Researcher Error: {research_state['error']}"}), 500

        research_summary = research_state.get("research_summary")

        blog_state = blogger.run(video_analysis, research_summary)
        
        if blog_state.get("error"):
            return jsonify({"error": f"Blogger Error: {blog_state['error']}"}), 500

        blog_post = blog_state.get("blog_post")

        return jsonify({
            "status": "success",
            "blog_post": blog_post,
            "debug_analysis": video_analysis,
            "debug_research": research_summary
        })

    except Exception as e:
        raise CustomException(e, sys)

if __name__ == '__main__':
    app.run(debug=True)