from utils.state import ResearchState, ChatState
from graph.workflow import create_research_graph, create_chat_graph

def run_research_assistant(topic: str):
    """Main function to run the research assistant"""
    
    print(f"🚀 Starting research on: {topic}")
    print("=" * 50)
    
    # Create initial state
    initial_state = ResearchState(
        topic=topic,
        subtopics=[],
        research_results={},
        final_report="",
        current_subtopic="",
        step_count=0,
        conversation_history=[]
    )
    
    app = create_research_graph()
    

    final_state = app.invoke(initial_state)
    
    print("\n" + "=" * 50)
    print("🎉 Research Complete!")
    print(f"📊 Steps completed: {final_state['step_count']}")
    print(f"📝 Subtopics researched: {len(final_state['research_results'])}")
    
    return final_state

if __name__ == "__main__":
    mode = input("Choose mode: [1] Research Report  [2] RAG Chatbot: ").strip()
    if mode == "2":
        app = create_chat_graph()
        state: ChatState = {
            "messages": [],
            "rephrased_query": "",
            "retrieved_context": "",
            "answer": "",
        }
        print("Type 'exit' to quit.")
        while True:
            q = input("You: ")
            if q.lower() in {"exit", "quit"}:
                break
            state["messages"].append({"role": "user", "content": q})
            state = app.invoke(state)
            print(f"Assistant: {state['answer']}")
    else:
        research_topic = input("Enter the research topic: ")
        result = run_research_assistant(research_topic)
        print("\n📄 Final Report (excerpt):")
        print("-" * 30)
        print(result["final_report"][:500] + "...")
        print("\n🔄 Conversation History:")
        for i, step in enumerate(result["conversation_history"], 1):
            print(f"{i}. {step}")