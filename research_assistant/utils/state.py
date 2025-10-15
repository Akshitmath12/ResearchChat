from typing import List, Dict, Any, TypedDict

class ResearchState(TypedDict):
    """State schema for our research workflow"""
    topic: str
    subtopics: List[str]
    research_results: Dict[str, str]
    final_report: str
    current_subtopic: str
    step_count: int
    conversation_history: List[str] 

class ChatState(TypedDict):
    """State schema for RAG chatbot workflow"""
    messages: List[Dict[str, str]]  # {role, content}
    rephrased_query: str
    retrieved_context: str
    answer: str