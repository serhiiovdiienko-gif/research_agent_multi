import wikipedia
import arxiv
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.agents.llm_agent import LlmAgent


def wikipedia_tool(query: str) -> str:
    """
    Searches Wikipedia for a given query and returns a summary of the top result.

    Args:
        query (str): The search term to look up on Wikipedia.
    """
    try:
        # The summary method automatically finds the best-matching page
        # and returns a summary of it.
        summary = wikipedia.summary(query)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        # Handle cases where a query is ambiguous (e.g., "Java")
        return f"The query '{query}' is ambiguous. Please be more specific. Options: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        # Handle cases where the page does not exist
        return f"Sorry, I could not find a Wikipedia page for '{query}'."
    except Exception as e:
        return f"An unexpected error occurred while searching Wikipedia: {e}"


def arxiv_tool(query: str) -> str:
    """
    Searches the arXiv repository for academic papers matching a query.

    Args:
        query (str): The topic to search for academic papers on.
    """
    try:
        # Create a client to interact with the arXiv API
        client = arxiv.Client()

        # Define the search parameters
        search = arxiv.Search(
            query=query,
            max_results=2,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        # Use the client to execute the search and get the results
        for result in client.results(search):
            results.append(f"Title: {result.title}\nSummary: {result.summary}\nURL: {result.entry_id}")
            
        if not results:
            return f"No academic papers found on arXiv for the query '{query}'."
            
        return "\n---\n".join(results)
        
    except Exception as e:
        return f"An unexpected error occurred while searching arXiv: {e}"
        
def report_writer_tool(content: str, filename: str) -> str:
    """
    Writes the given content to a local file. Appends if the file already exists.

    Args:
        content (str): The text content to write to the file.
        filename (str): The name of the file to save the content in (e.g., 'report.txt').
    """
    try:
        # Use 'a' for append mode. This will create the file if it doesn't exist,
        # or add to the end of it if it does.
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(content + "\n")
        return f"Successfully appended content to {filename}."
    except Exception as e:
        return f"An error occurred while writing to file: {e}"
    
# Define a constant for the model name to be used by all agents
MODEL_NAME = 'gemini-1.5-flash'
#MODEL_NAME = 'gemini-2.5-flash-lite'

# Define the wikipedia agent

wikipedia_agent = LlmAgent(
    name='wikipedia_researcher',
    model=MODEL_NAME,
    description='An expert at finding and summarizing information from Wikipedia.',
    instruction='You are a specialized agent and your only task is to extract the research TOPIC from the request and use the `wikipedia_tool` to find relevant information.',
    tools=[wikipedia_tool],
    output_key="wikipedia_notes"
)

# Define the arxiv agent

arxiv_agent = LlmAgent(
    name='arxiv_researcher',
    model=MODEL_NAME,
    description='An expert at finding and summarizing academic papers from the arXiv repository.',
    instruction='You are a specialized agent whose only job is to search arXiv for academic papers on a given topic. Use the arxiv_tool to find them.',
    tools=[arxiv_tool]
)

# Define the Google search agent

google_search_agent = LlmAgent(
    name='web_searcher',
    model=MODEL_NAME,
    description='An expert at searching the web to find relevant, up-to-date information on a topic using Google Search.',
    instruction="""Your only task is to accept a research topic, use the `GoogleSearchTool` to find information about it, and return a concise summary of the search results.""",
    tools=[GoogleSearchTool(bypass_multi_tools_limit=True)],
)

# Define the writer agent

writer_agent = LlmAgent(
    name='report_writer',
    model='gemini-2.5-flash-lite',
    description='An expert at writing content to a file.',
    instruction=(
        "You are a specialized writing agent.\n"
        "Write a short report based on the user's request and the research notes below.\n\n"
        "Wikipedia notes:\n{wikipedia_notes}\n\n"
        "Then save the final report to text file using `report_writer_tool`."
        "The filename should be based on the research topic (e.g., black_holes_report.txt)..\n"
    ),    
    tools=[report_writer_tool]
)

# Define the system prompt for our controler agent

controller_instruction = """
You are a research assistant who orchestrates a team of specialist agents to produce a high-quality research report. 
Your primary role is to delegate tasks, synthesize the results, and ensure the final report is well-structured.

Your specialist team consists of:
- `wikipedia_researcher`: Use this agent to get general background information and a high-level overview.
- `arxiv_researcher`: Use this agent to find relevant academic papers and their summaries.
- `web_searcher`: Use this agent to find up-to-date information and supplementary context from the web.

Your workflow must be as follows:
1.  First, call all three specialist research agents (`wikipedia_researcher`, `arxiv_researcher`, and `web_searcher`) to gather a comprehensive set of information on the topic.
2.  Once all information has been gathered, you must personally synthesize the content from all three sources into a single, coherent summary.
3.  Finally, call the `report_writer` tool to save the complete, synthesized report into a text file. The filename should be based on the research topic (e.g., black_holes_report.txt).
"""

# Define the controler agent

root_agent = LlmAgent(
    name='controller',
    model=MODEL_NAME,
    description='The main controller for a multi-agent research team.',
    instruction=controller_instruction,
    tools=[
        AgentTool(wikipedia_agent),
        AgentTool(arxiv_agent),
        AgentTool(google_search_agent),
        AgentTool(writer_agent)
    ],
)