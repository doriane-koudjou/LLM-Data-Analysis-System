"""
analyst_agent.py
----------------
Central LangChain agent for automated data analysis.
Supports both single and multiple database connections.

Migration note (Fix 3):
  Switched from create_react_agent (text-based ReAct) to
  create_tool_calling_agent (native API function calling).

  Why:
  - create_react_agent requires the LLM to write "Action: <tool>"
    as plain text which a strict parser then reads back.  Modern
    models (GPT-4o, Claude 3.x) are trained to call tools via their
    native API and can produce subtle formatting deviations that
    break the parser, causing an infinite Thought loop until the
    iteration limit is reached.
  - create_tool_calling_agent dispatches tools through the model's
    built-in function-calling interface (JSON schema), eliminating
    the parsing step entirely.  Both Claude and GPT-4o support this
    natively, making the agent compatible with either model without
    prompt-format adjustments.

  Changes:
  - create_react_agent  →  create_tool_calling_agent
  - PromptTemplate      →  ChatPromptTemplate with system / placeholder
  - max_iterations 10   →  15  (aligned with evaluation suite)
  - ReAct FORMAT block removed from prompt (no longer needed)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from database.db_connector import DatabaseConnector
from database.multi_db_connector import MultiDatabaseConnector
from tools.sql_tool import get_sql_tools
from tools.multi_db_sql_tool import get_multi_db_tools
from tools.python_tool import get_python_tool
from tools.predictive_tool import get_predictive_tool
from utils.prompt_templates import get_full_system_prompt

load_dotenv(Path(__file__).parent.parent / '.env')

SYSTEM_PROMPT = get_full_system_prompt()


def get_llm():
    """Initializes the LLM based on LLM_PROVIDER env variable."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=4096,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            max_tokens=4096,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Use 'anthropic' or 'openai'."
        )


class DataAnalystAgent:
    """
    Tool-Calling LangChain agent for automated data analysis.
    Supports single DatabaseConnector or MultiDatabaseConnector.
    """

    def __init__(self, connector: DatabaseConnector | MultiDatabaseConnector):
        self.connector = connector
        self.llm = get_llm()
        self.tools = self._build_tools()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,   # must be True for ChatPromptTemplate
        )
        self.agent_executor = self._build_agent()

    def _build_tools(self) -> list:
        """
        Builds the tool set based on connector type.
        MultiDatabaseConnector gets multi-db tools,
        single DatabaseConnector gets standard tools.
        """
        python_tool = get_python_tool()

        if isinstance(self.connector, MultiDatabaseConnector):
            schema_tool, sql_tool, list_tool = get_multi_db_tools(self.connector)
            first_name = self.connector.get_connection_names()[0]
            first_connector = self.connector.get_connector(first_name)
            pred_tool = get_predictive_tool(first_connector)
            return [list_tool, schema_tool, sql_tool, python_tool, pred_tool]
        else:
            sql_tool, schema_tool = get_sql_tools(self.connector)
            pred_tool = get_predictive_tool(self.connector)
            return [schema_tool, sql_tool, python_tool, pred_tool]

    def _build_agent(self) -> AgentExecutor:
        """Builds the Tool-Calling agent with memory and tools."""
        # ChatPromptTemplate replaces the old PromptTemplate.
        # MessagesPlaceholder("chat_history") carries the conversation memory.
        # MessagesPlaceholder("agent_scratchpad") carries the tool-call loop.
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
        )

    def run(self, question: str) -> str:
        """Runs the agent and returns the response."""
        try:
            print(f"[DEBUG] Agent.run called with: {question}", flush=True)
            response = self.agent_executor.invoke({"input": question})
            print(f"[DEBUG] Agent.run finished", flush=True)
            return response.get("output", "No response generated.")
        except Exception as e:
            print(f"[DEBUG] Agent.run error: {e}", flush=True)
            return f"Agent error: {e}"
    def reset_memory(self):
        """Clears the conversation history."""
        self.memory.clear()