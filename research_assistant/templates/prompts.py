from langchain.prompts import PromptTemplate, ChatPromptTemplate

subtopic_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
    Break down the research topic "{topic}" into 3-4 key subtopics that would provide 
    comprehensive coverage of the subject.
    
    Return only the subtopics as a numbered list:
    1. [Subtopic 1]
    2. [Subtopic 2]
    3. [Subtopic 3]
    4. [Subtopic 4]
    """
)

# RAG chatbot prompts
rephrase_prompt = PromptTemplate(
    input_variables=["query", "history"],
    template="""
    You are a helpful assistant. Given the chat history and a follow-up user query,
    rewrite the query to be standalone and unambiguous.

    Chat history:
    {history}

    Follow-up query:
    {query}

    Standalone rephrased query:
    """
)

rag_answer_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    Use the provided context to answer the user's question. If the answer is not
    contained in the context, say you don't know.

    Context:
    {context}

    Question:
    {question}

    Provide a concise, factual answer and include inline citations like [1], [2] matching the context items.
    """
)

research_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a thorough researcher. Provide detailed, accurate information."),
    ("human", """
    Research the following subtopic: {subtopic}
    
    Main topic context: {main_topic}
    
    Provide a comprehensive overview including:
    - Key concepts and definitions
    - Important facts and statistics
    - Current trends or developments
    - Practical implications
    
    Keep it informative but concise (2-3 paragraphs).
    """)
])

# Template for final report compilation
report_prompt = PromptTemplate(   
    input_variables=["topic", "research_data"],  
    template="""  #template for the report
    Create a comprehensive research report on "{topic}" using the following research data:
    
    {research_data}
    
    Structure the report with:
    1. Executive Summary
    2. Key Findings (organized by subtopic)
    3. Conclusions and Implications
    4. Recommendations for further research
    
    Make it professional and well-organized.
    """
)