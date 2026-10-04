from crewai import Task

def create_research_task(agent, question: str) -> Task:
    return Task(
        description=f"""Research the question thoroghly:
        Question: {question}

Use your available tools to find relevant evidence:
- Use vector_search for annual report data or financial terminology
- Use sql_query for stock prices, crypto prices, or market data

Return all relevant evidence you found including sources.""",

        expected_output="""A detailed collection of relevant financial evidence including:
- The actual data or text found
- The source of each piece of information (company name, year, table name)
- Any important caveats (data not available, outside date range, company not in database)""",
        agent=agent
)
    
def create_analysis_task(agent, question: str, research_task: Task) -> Task:
    return Task(
        description=f"""Analyze the rsearch evidence provide and write a clear, accurate and well-sourced answer to the question.
        Question: {question}

Base your answer strictly on the evidence. Always cite your sources.
If the evidence indicates data is unavailable, communicate that clearly and helpfully.""",

        expected_output="""A well-structured answer that:
- Directly addresses the user's question
- Cites sources (company name, report year, or data table)
- Is honest about any data limitations
- Is written in clear, professional financial language""",
        agent=agent,
        context=[research_task]
)