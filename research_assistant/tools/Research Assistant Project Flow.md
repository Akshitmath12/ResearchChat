Research Assistant Project Flow
This is a LangGraph-based research assistant that automates the process of researching any topic by breaking it down into subtopics and generating comprehensive reports. Here's how it works:
🏗️ Architecture Overview
The project follows a state-based workflow using LangGraph, which orchestrates different research nodes in a structured manner.
📋 Core Components
State Management (utils/state.py)
Defines ResearchState with fields: topic, subtopics, research_results, final_report, etc.
Tracks progress through the research workflow
Workflow Graph (graph/workflow.py)
Creates a StateGraph that defines the research flow
Manages conditional branching and node transitions
Research Nodes (graph/nodes.py)
Four main processing nodes that handle different research phases
LLM Chains (chains/research_chains.py)
Handles different LLM providers (Ollama, HuggingFace, OpenAI)
Creates specialized chains for different research tasks
Tools (tools/research_tools.py)
Web search simulation and document saving utilities
🔄 Workflow Flow
history
�� Key Features
Multi-LLM Support:
Ollama (Free, local) - Default
HuggingFace (Free, limited)
OpenAI (Paid, high-quality)
Modular Design:
Each research phase is a separate node
Easy to modify or extend individual components
State Tracking:
Maintains conversation history
Tracks step count and progress
Stores intermediate results
Flexible Configuration:
Environment-based settings
Configurable max subtopics
Adjustable temperature for LLM responses
📁 File Structure Breakdown
main.py: Entry point, initializes state and runs the workflow
graph/: Workflow orchestration and node definitions
chains/: LLM chain configurations and providers
tools/: Utility functions for search and file operations
templates/: Prompt templates for different research phases
utils/: State schema definition
outputs/: Generated research reports (timestamped)
🚀 Usage Flow
User runs python main.py
Enters a research topic
System automatically:
Analyzes topic → generates subtopics
Researches each subtopic sequentially
Compiles comprehensive report
Saves results to timestamped file
💡 Example Output
Based on the existing outputs, the system generates detailed research reports like:
research_report_Artificial_Intelligence_in_Healthcare_20250919_111227.txt
research_report_Cricket_20250919_154238.txt
This is a well-structured, production-ready research automation tool that leverages modern LangChain and LangGraph frameworks to create an intelligent research assistant!
