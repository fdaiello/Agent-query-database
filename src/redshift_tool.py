import os
from dotenv import load_dotenv
import boto3
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

# --- Step 1: Load environment variables ---
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
REDSHIFT_WORKGROUP_NAME = os.getenv("REDSHIFT_WORKGROUP_NAME")
REDSHIFT_DATABASE = os.getenv("REDSHIFT_DATABASE")
# For serverless, DbUser is not required

if not all([AWS_REGION, REDSHIFT_WORKGROUP_NAME, REDSHIFT_DATABASE]):
    raise ValueError("Missing AWS Redshift Serverless environment variables in .env")

# --- Step 2: Create boto3 Redshift Data API client ---
redshift_client = boto3.client("redshift-data", region_name=AWS_REGION)

# --- Step 3: Define a LangChain tool to run queries ---
@tool
def query_redshift(sql: str) -> str:
    """
    Run a SQL query against AWS Redshift Serverless using the Data API and return results as a string.
    """
    try:
        # Submit query
        res = redshift_client.execute_statement(
            WorkgroupName=REDSHIFT_WORKGROUP_NAME,
            Database=REDSHIFT_DATABASE,
            Sql=sql
        )
        query_id = res["Id"]

        # Wait for completion
        while True:
            status = redshift_client.describe_statement(Id=query_id)
            if status["Status"] in ["FINISHED", "FAILED", "ABORTED"]:
                break

        if status["Status"] != "FINISHED":
            return f"Query failed: {status.get('Error', 'Unknown error')}"

        # Fetch results
        result = redshift_client.get_statement_result(Id=query_id)
        # Format nicely
        columns = [col["name"] for col in result["ColumnMetadata"]]
        rows = [
            dict(zip(columns, [v.get("stringValue", "") for v in row]))
            for row in result["Records"]
        ]
        return str(rows)

    except Exception as e:
        return f"Error running query: {str(e)}"

# --- Step 4: LLM configuration ---
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# --- Step 5: Agent with Redshift tool ---
agent_executor = initialize_agent(
    tools=[query_redshift],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# --- Step 6: Interactive query loop ---
print("💬 Ask me questions about your Redshift Serverless database! (type 'exit' to quit)")
while True:
    user_input = input("\n❓ Your question: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break
    try:
        response = agent_executor.run(user_input)
        print("\n📊 Answer:", response)
    except Exception as e:
        print("⚠️ Error:", str(e))