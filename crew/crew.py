import os
os.environ["CREWAI_DISABLE_PROMPT_CACHING"] = "true"

# Monkey patch to fix CrewAI sending cache_breakpoint to Groq
import litellm
original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        cleaned = []
        for msg in kwargs["messages"]:
            cleaned.append({k: v for k, v in msg.items() if k != "cache_breakpoint"})
        kwargs["messages"] = cleaned
    return original_completion(*args, **kwargs)

litellm.completion = patched_completion

from dotenv import load_dotenv
from crewai import Crew, Process, LLM
from crew.agents import create_researcher, create_analyst
from crew.tasks import create_research_task, create_analysis_task

load_dotenv()

def run(question: str, model: str = None, api_key: str =None) -> str:
    """
    Main entry point called by app.py.
    model and api_key are always passed from app.py.
    crew.py has no fallback logic — that is app.py's responsibility.
    """
    
    # Create LLM
    if model.startswith("ollama/"):
        llm = LLM(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.6
        )
    else:
        if not api_key:
            return (
                "No API key provided. "
                "Please enter your API key in the sidebar "
                "or switch to Ollama for free local inference."
            )
        llm = LLM(
            model=model,
            api_key=api_key,
            temperature=0.6
        )
        
    # Create Agents
    researcher = create_researcher(llm)
    analyst = create_analyst(llm)
    
    # Create Tasks
    research_task = create_research_task(researcher, question)
    analysis_task = create_analysis_task(analyst, question, research_task)
    
    # Build and run Crew
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        process=Process.sequential,
        verbose=True
    )
    
    return str(crew.kickoff())

if __name__ == "__main__":
    test_question = "What was Apple's revenue in 2023?"
    print(f"Question: {test_question}")
    answer = run(
        question=test_question,
        model="groq/openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )
    print(f"\nAnswer: \n{answer}")