import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent

# Load .env for API keys and DB credentials
load_dotenv()

# --- Step 1: Configure database connection ---
db_uri = os.getenv("DB_URI")
if not db_uri:
    raise ValueError("DB_URI not set in .env file")

db = SQLDatabase.from_uri(db_uri)

# --- Step 2: Create LLM and Tool ---
llm = ChatOpenAI(
    model="gpt-4o",  # or gpt-4, gpt-3.5-turbo
    temperature=0
)

def sql_query_tool_func(query: str):
    """Executes SQL against the configured database and returns results as a list of dicts."""
    return str(db.run(query))

sql_tool = Tool(
    name="sql_database",  # Changed from "SQL Database" to a valid name
    func=sql_query_tool_func,
    description="Executes SQL against the configured database and returns results as a list of dicts."
)

tools = [sql_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can answer questions by querying a SQL database."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("intermediate_steps"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt,
)

print("💬 Ask me questions about your database! (type 'exit' to quit)")
while True:
    user_input = input("\n❓ Your question: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    try:
        response = agent.invoke({
            "input": user_input,
            "chat_history": [],
            "intermediate_steps": [],
            "agent_scratchpad": []
        }, return_only_outputs=True)
        # If response is a dict with 'output', print that, else print response
        if isinstance(response, dict) and 'output' in response:
            print("\n📊 Answer:", response['output'])
        else:
            print("\n📊 Answer:", response)
    except Exception as e:
        print("⚠️ Error:", str(e))