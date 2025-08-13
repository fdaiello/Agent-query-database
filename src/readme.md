# Agent Versions Overview

This directory contains several versions of a GenAI-powered agent for querying databases using natural language. Each version demonstrates a different approach or backend, suitable for various environments and requirements.

## Agent Variants

### 1. `init_demo_db.py`
- **Purpose:** Initializes a local SQLite demo database with sample tables and data.
- **Usage:** Run this script to create and populate `example.db` for local testing and development.
- **Database:** SQLite (local file)

### 2. `jdbc_generic.py`
- **Purpose:** Generic agent for any SQL database accessible via JDBC or Python DB-API.
- **Backend:** Uses LangChain's `SQLDatabase` and `SQLDatabaseToolkit` for schema introspection and query execution.
- **Configuration:** Set `DB_URI` in `.env` to your database connection string (e.g., SQLite, PostgreSQL, MySQL, etc.).
- **Usage:** Run the script and interactively ask questions; the agent translates them to SQL and executes them.
- **Best for:** Local databases, cloud databases with direct access, or any DB supported by SQLAlchemy.

### 3. `redshift_tool.py`
- **Purpose:** Minimal agent for AWS Redshift Serverless using the Redshift Data API.
- **Backend:** Uses the Data API (HTTP) to connect to Redshift Serverless, even inside a VPC.
- **Features:** Defines a LangChain tool for SQL execution, integrates with LangChain agent.
- **Configuration:** Set AWS and Redshift variables in `.env`.
- **Best for:** Redshift Serverless, cloud-native, serverless, or VPC environments.

### 4. `sql_agent_rda.py`
- **Purpose:** Advanced agent for Redshift Serverless using the Data API, with schema introspection and conversational memory.
- **Backend:** Uses Redshift Data API for SQL execution and schema fetching.
- **Features:**
  - Fetches schema info from Redshift to enrich LLM prompt context.
  - Maintains conversation history for multi-turn interactions.
  - Modular utility functions in `redshift_utils.py`.
- **Best for:** Redshift Serverless, production, or cloud-native deployments.

### 5. `sql_agent.py`
- **Purpose:** Advanced agent for any SQL database using LangChain's SQLDatabase and LangGraph workflow.
- **Backend:** Uses SQLDatabase for schema and query execution.
- **Features:**
  - Schema introspection for prompt enrichment.
  - Conversational memory and multi-step workflow.
- **Best for:** Any database supported by SQLAlchemy, local or remote.

## How to Choose
- **Local/SQLite:** Use `init_demo_db.py` and `jdbc_generic.py` or `sql_agent.py`.
- **Redshift Serverless (VPC/Cloud):** Use `redshift_tool.py` for minimal, or `sql_agent_rda.py` for advanced features.
- **Other Databases:** Use `jdbc_generic.py` or `sql_agent.py` and set `DB_URI` accordingly.

## Environment Setup
- All agents require a `.env` file with relevant credentials and connection info.
- For Redshift Data API, set AWS credentials and Redshift config.
- For JDBC/SQLDatabase, set `DB_URI`.

## Extensibility
- You can adapt any agent to other databases by swapping out the backend connection and schema introspection logic.
- The Redshift Data API approach is ideal for serverless/cloud, while SQLDatabase is best for direct DB access.

## See Also
- `redshift_utils.py`: Utility functions for Redshift Data API.
- `README.md`: Project overview and setup instructions.

---
MIT License
