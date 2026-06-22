from typing import TypedDict
import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    stdio_client,
    StdioServerParameters
)

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()
root = Path(__file__).resolve().parent
load_dotenv(root / ".env")
load_dotenv(root / "uploads" / ".env", override=False)


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)


class GraphState(TypedDict):
    question: str
    route: str
    answer: str


async def call_tool(
    tool_name,
    arguments
):

    server = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server) as (
        read_stream,
        write_stream
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments
            )

            return str(result)


async def read_resource(uri):

    server = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server) as (
        read_stream,
        write_stream
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            result = await session.read_resource(
                uri
            )

            return str(result)

async def get_prompt(name):

    server = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    async with stdio_client(server) as (
        read_stream,
        write_stream
    ):

        async with ClientSession(
            read_stream,
            write_stream
        ) as session:

            await session.initialize()

            result = await session.get_prompt(
                name
            )

            return str(result)


def agent(state):

    question = state["question"]

    prompt = f"""
You are a routing agent.

Available options:

tool
resource
prompt

Rules:

- Mathematical calculations -> tool

- If user asks to summarize, rewrite,
  create notes, shorten, simplify,
  or translate a document -> prompt

- Any question asking information
  from the uploaded PDF -> resource

Question:
{question}

Return only one word.
"""

    response = llm.invoke(prompt)

    return {
        "route": response.content.strip().lower()
    }

def route_after_agent(state):
    return state["route"]

def tool_node(state):

    result = asyncio.run(
        call_tool(
            "calculator",
            {
                "expression": state["question"]
            }
        )
    )

    return {
        "answer": result
    }

def resource_node(state):

    context = asyncio.run(
        read_resource(
            f"pdf://search/{state['question']}"
        )
    )

    response = llm.invoke(
        f"""
        Context:
        {context}

        Question:
        {state['question']}

        Answer only from context.
        """
    )

    return {
        "answer": response.content
    }

def prompt_node(state):

    prompt_text = asyncio.run(
        get_prompt(
            "summarize_pdf"
        )
    )

    policy = asyncio.run(
        read_resource(
            f"pdf://search/{state['question']}"
        )
    )

    response = llm.invoke(
        f"""
        {prompt_text}

        Content:

        {policy}
    """
    )

    return {
        "answer": response.content
    }


from langgraph.graph import (
    StateGraph,
    END
)

builder = StateGraph(GraphState)

builder.add_node(
    "agent",
    agent
)

builder.add_node(
    "tool_node",
    tool_node
)

builder.add_node(
    "resource_node",
    resource_node
)

builder.add_node(
    "prompt_node",
    prompt_node
)

builder.set_entry_point(
    "agent"
)

builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tool": "tool_node",
        "resource": "resource_node",
        "prompt": "prompt_node"
    }
)

builder.add_edge(
    "tool_node",
    END
)

builder.add_edge(
    "resource_node",
    END
)

builder.add_edge(
    "prompt_node",
    END
)

graph = builder.compile()