import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import Tool
import boto3

# Redshift Data API config
REDSHIFT_CLUSTER_ID = os.getenv("REDSHIFT_CLUSTER_ID")
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE")
REDSHIFT_DB_USER = os.getenv("REDSHIFT_DB_USER")
REDSHIFT_WORKGROUP_NAME = os.getenv("REDSHIFT_WORKGROUP_NAME")
AWS_REGION = os.getenv("AWS_REGION")

class RedshiftDataAPIWrapper:
    """Minimal wrapper for Redshift Data API for LangChain agent compatibility. Supports provisioned and serverless."""
    def __init__(self):
        self.client = boto3.client("redshift-data", region_name=AWS_REGION)
    def run(self, query: str):
        kwargs = {
            "Database": REDSHIFT_DATABASE,
            "Sql": query
        }
        if REDSHIFT_WORKGROUP_NAME:
            kwargs["WorkgroupName"] = REDSHIFT_WORKGROUP_NAME
        else:
            kwargs["ClusterIdentifier"] = REDSHIFT_CLUSTER_ID
            kwargs["DbUser"] = REDSHIFT_DB_USER
        resp = self.client.execute_statement(**kwargs)
        qid = resp["Id"]
        # Wait for completion
        while True:
            desc = self.client.describe_statement(Id=qid)
            if desc["Status"] in ("FINISHED", "FAILED", "ABORTED"):
                break
        if desc["Status"] != "FINISHED":
            raise RuntimeError(f"Query failed: {desc.get('Error', 'Unknown')}")
        result_resp = self.client.get_statement_result(Id=qid)
        columns = [c["name"] for c in result_resp["ColumnMetadata"]]
        results = []
        for rec in result_resp["Records"]:
            row = {}
            for col, val in zip(columns, rec):
                row[col] = list(val.values())[0] if val else None
            results.append(row)
        return results

# Try loading .env (safe for local dev)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set in environment variables.")

llm = OpenAI(temperature=0, api_key=openai_api_key)

db = RedshiftDataAPIWrapper()

# Define the Redshift tool for the agent
redshift_tool = Tool(
    name="Redshift SQL",
    func=lambda q: str(db.run(q)),
    description="Executes SQL against AWS Redshift via Data API and returns results as a list of dicts."
)

tools = [redshift_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that can answer questions by querying an AWS Redshift database."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt,
)

def ask_question(question: str):
    """Ask a natural language question and get the answer from the database."""
    messages = [HumanMessage(content=question)]
    return agent.invoke({"input": question, "chat_history": [], "agent_scratchpad": []})

if __name__ == "__main__":
    user_question = input("Ask a question about your data: ")
    answer = ask_question(user_question)
    print("Answer:", answer)