import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent  # ✅ Updated import
from langchain.agents.agent_types import AgentType  # still valid

# Load .env for API keys and DB credentials
load_dotenv()

# --- Step 1: Configure database connection ---
# Example for PostgreSQL:
# Format: postgresql+psycopg2://username:password@host:port/dbname
db_uri = os.getenv("DB_URI")
if not db_uri:
    raise ValueError("DB_URI not set in .env file")

db = SQLDatabase.from_uri(db_uri)

# --- Step 2: Create LLM and Toolkit ---
llm = ChatOpenAI(
    model="gpt-4o",  # or gpt-4, gpt-3.5-turbo
    temperature=0
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# --- Step 3: Create SQL Agent ---
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Step 4: Interactive query loop ---
print("💬 Ask me questions about your database! (type 'exit' to quit)")
while True:
    user_input = input("\n❓ Your question: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break

    try:
        response = agent_executor.run(user_input)
        print("\n📊 Answer:", response)
    except Exception as e:
        print("⚠️ Error:", str(e))