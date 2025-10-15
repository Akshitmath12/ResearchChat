from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from templates.prompts import (
    subtopic_prompt,
    research_prompt,
    report_prompt,
    rephrase_prompt,
    rag_answer_prompt,
)
from config import (
    LLM_PROVIDER, 
    OLLAMA_MODEL, 
    OLLAMA_BASE_URL,
    OPENAI_API_KEY, 
    HUGGINGFACE_API_KEY,
    TEMPERATURE
)

from config import (LLM_PROVIDER, TEMPERATURE, OPENAI_API_KEY, OLLAMA_MODEL, OLLAMA_BASE_URL)

def get_llm():
    """Get the appropriate LLM based on configuration"""
    
    if LLM_PROVIDER == "ollama":
        print("🦙 Using Ollama (Free Local LLM)")
        return OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE
        )
    
    elif LLM_PROVIDER == "huggingface":
        print("🤗 Using Hugging Face (Free)")
        try:
            from transformers import pipeline
            # Use a free model from Hugging Face
            pipe = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",  # Free model
                max_length=1000,
                temperature=TEMPERATURE
            )
            return HuggingFacePipeline(pipeline=pipe)
        except ImportError:
            print("❌ transformers not installed. Installing...")
            print("Run: pip install transformers torch")
            return None
    
    elif LLM_PROVIDER == "openai":
        print("🤖 Using OpenAI (Paid)")
        if not OPENAI_API_KEY:
            print("❌ OpenAI API key not found!")
            return None
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=TEMPERATURE,
            api_key=OPENAI_API_KEY
        )
    
    else:
        print(f"❌ Unknown LLM provider: {LLM_PROVIDER}")
        return None

# Initialize LLM
llm = get_llm()

if llm is None:
    print("❌ Failed to initialize LLM. Please check your configuration.")
    exit(1)

# Modern LangChain approach - using RunnableSequence instead of LLMChain
def create_chains():
    """Create chains using modern LangChain syntax"""
    
    # Chain for breaking down topics (Modern syntax)
    subtopic_chain = subtopic_prompt | llm
    
    # Chain for individual research  
    research_chain = research_prompt | llm
    
    # Chain for final report
    report_chain = report_prompt | llm

    # RAG chains
    rephrase_chain = rephrase_prompt | llm
    rag_answer_chain = rag_answer_prompt | llm
    
    return subtopic_chain, research_chain, report_chain, rephrase_chain, rag_answer_chain

# Create the chains
subtopic_chain, research_chain, report_chain, rephrase_chain, rag_answer_chain = create_chains()