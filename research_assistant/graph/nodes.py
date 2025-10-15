from utils.state import ResearchState, ChatState
from chains.research_chains import subtopic_chain, research_chain, report_chain, rephrase_chain, rag_answer_chain
from tools.research_tools import save_document_tool, get_retriever
from config import MAX_SUBTOPICS
from datetime import datetime
from langchain_core.runnables import RunnablePassthrough

def analyze_topic_node(state: ResearchState) -> ResearchState:
    """Node 1: Break down main topic into subtopics"""
    print(f"🔍 Analyzing topic: {state['topic']}")
    
    # Use modern chain syntax
    result = subtopic_chain.invoke({"topic": state["topic"]})
    
    # Parse subtopics from the result
    subtopics = []
    lines = result.split('\n') if isinstance(result, str) else str(result).split('\n')
    
    for line in lines:
        if line.strip() and any(char.isdigit() for char in line):
           
            subtopic = line.split('.', 1)[1].strip() if '.' in line else line.strip()
            if subtopic:
                subtopics.append(subtopic)
    
    state["subtopics"] = subtopics[:MAX_SUBTOPICS]
    state["step_count"] = state.get("step_count", 0) + 1
    state["conversation_history"].append(f"Generated {len(subtopics)} subtopics")
    
    print(f"📝 Generated subtopics: {subtopics}")
    return state

def research_subtopic_node(state: ResearchState) -> ResearchState:
    """Node 2: Research individual subtopics"""
    if not state["subtopics"]:
        return state
    
    # Get next subtopic to research
    current_subtopic = state["subtopics"][0]
    state["current_subtopic"] = current_subtopic
    
    print(f"📚 Researching: {current_subtopic}")
    
    # Use modern chain syntax
    research_result = research_chain.invoke({
        "subtopic": current_subtopic,
        "main_topic": state["topic"]
    })
    
    # Store result
    if "research_results" not in state:
        state["research_results"] = {}
    
    # Convert result to string if needed
    result_text = research_result if isinstance(research_result, str) else str(research_result)
    state["research_results"][current_subtopic] = result_text
    
    # Remove researched subtopic from list
    state["subtopics"] = state["subtopics"][1:]
    state["step_count"] = state.get("step_count", 0) + 1
    state["conversation_history"].append(f"Researched: {current_subtopic}")
    
    return state

def compile_report_node(state: ResearchState) -> ResearchState:
    """Node 3: Compile final research report"""
    print("📄 Compiling final report...")
    
    # Format research data for report
    research_data = ""
    for subtopic, result in state["research_results"].items():
        research_data += f"\n## {subtopic}\n{result}\n"
    
    # Generate final report using modern chain syntax
    final_report = report_chain.invoke({
        "topic": state["topic"],
        "research_data": research_data
    })
    
    # Convert to string if needed
    state["final_report"] = final_report if isinstance(final_report, str) else str(final_report)
    state["step_count"] = state.get("step_count", 0) + 1
    state["conversation_history"].append("Compiled final report")
    
    return state

def save_results_node(state: ResearchState) -> ResearchState:
    """Node 4: Save results to file"""
    print("💾 Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{state['topic'].replace(' ', '_')}_{timestamp}"
    
    # Use document tool to save
    save_result = save_document_tool(state["final_report"], filename)
    
    state["conversation_history"].append(f"Saved report: {save_result}")
    print(f"✅ {save_result}")
    
    return state

# ------------------ RAG Chatbot Nodes -----------------

def rephrase_query_node(state: ChatState) -> ChatState:
    """Rewrite the latest user query to standalone using history"""
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in state.get("messages", [])[:-1]])
    user_query = state.get("messages", [])[-1]["content"] if state.get("messages") else ""
    rephrased = rephrase_chain.invoke({"query": user_query, "history": history_text})
    state["rephrased_query"] = rephrased if isinstance(rephrased, str) else str(rephrased)
    return state

def retrieve_context_node(state: ChatState) -> ChatState:
    """Retrieve similar chunks for the rephrased query"""
    retriever = get_retriever()
    docs = retriever.invoke(state.get("rephrased_query", ""))
    context = "\n\n".join([f"[{i+1}] {d.page_content}\n(Source: {d.metadata.get('source','n/a')})" for i, d in enumerate(docs)])
    state["retrieved_context"] = context
    return state

def generate_answer_node(state: ChatState) -> ChatState:
    """Generate final answer with citations using retrieved context"""
    question = state.get("rephrased_query", "")
    context = state.get("retrieved_context", "")
    answer = rag_answer_chain.invoke({"question": question, "context": context})
    state["answer"] = answer if isinstance(answer, str) else str(answer)
    return state

def append_message_node(state: ChatState) -> ChatState:
    """Append assistant message to history"""
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": state.get("answer", "")})
    state["messages"] = messages
    return state