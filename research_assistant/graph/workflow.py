from langgraph.graph import StateGraph, END # type: ignore
from utils.state import ResearchState, ChatState
from graph.nodes import (
    analyze_topic_node,
    research_subtopic_node, 
    compile_report_node,
    save_results_node
)
# Taken all the nodes means all the functions and chains as of what to do in each step
def should_continue_research(state: ResearchState) -> str:
    """Conditional edge: check if more subtopics need research"""
    if state["subtopics"]:  # Still have subtopics to research
        return "continue_research"
    else:  # All subtopics researched
        return "compile_report"
    

def create_research_graph():
    """Create the research workflow graph"""
    
    # Initialize StateGraph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("analyze_topic", analyze_topic_node)
    workflow.add_node("research_subtopic", research_subtopic_node)
    workflow.add_node("compile_report", compile_report_node)
    workflow.add_node("save_results", save_results_node)
    
    # Add edges
    workflow.set_entry_point("analyze_topic")
    
    # From analyze_topic, always go to research
    workflow.add_edge("analyze_topic", "research_subtopic")
    
    # From research_subtopic, conditional branching
    workflow.add_conditional_edges(
        "research_subtopic",
        should_continue_research,
        {
            "continue_research": "research_subtopic",  # Loop back for more research
            "compile_report": "compile_report"
        }
    )
    
    # From compile_report to save_results
    workflow.add_edge("compile_report", "save_results")
    
    # End after saving
    workflow.add_edge("save_results", END)
    
    return workflow.compile()

def create_chat_graph():
    """Create a simple RAG chatbot workflow graph"""
    from graph.nodes import (
        rephrase_query_node,
        retrieve_context_node,
        generate_answer_node,
        append_message_node,
    )

    chat = StateGraph(ChatState)
    chat.add_node("rephrase_query", rephrase_query_node)
    chat.add_node("retrieve_context", retrieve_context_node)
    chat.add_node("generate_answer", generate_answer_node)
    chat.add_node("append_message", append_message_node)

    chat.set_entry_point("rephrase_query")
    chat.add_edge("rephrase_query", "retrieve_context")
    chat.add_edge("retrieve_context", "generate_answer")
    chat.add_edge("generate_answer", "append_message")
    chat.add_edge("append_message", END)

    return chat.compile()