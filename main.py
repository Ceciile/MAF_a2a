# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import datetime
import os
from random import randint
from typing import Annotated

from agent_framework import tool
from agent_framework.openai import OpenAIChatClient

from agent_framework.ollama import OllamaChatClient

"""
Ollama with OpenAI Chat Client Example

This sample demonstrates using Ollama models through OpenAI Chat Client by
configuring the base URL to point to your local Ollama server for local AI inference.
Ollama allows you to run large language models locally on your machine.

Environment Variables:
- OLLAMA_ENDPOINT: The base URL for your Ollama server (e.g., "http://localhost:11434/v1/")
- OLLAMA_MODEL: The model name to use (e.g., "mistral", "llama3.2", "phi3")
"""


# NOTE: approval_mode="never_require" is for sample brevity. Use "always_require" in production; see samples/getting_started/tools/function_tool_with_approval.py and samples/getting_started/tools/function_tool_with_approval_and_threads.py.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """Get the weather for a given location."""
    conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."

@tool(approval_mode="never_require")
def get_time(location: str) -> str:
    """Get the current time."""
    return f"The current time in {location} is {datetime.now().strftime('%I:%M %p')}."

time_agent = OllamaChatClient().as_agent(
    name="TimeAgent",
    instructions="You are a helpful time agent answer in one sentence.",
    tools=get_time,
)


async def non_streaming_example() -> None:
    """Example of non-streaming response (get the complete result at once)."""
    print("=== Non-streaming Response Example ===")

    # agent = OpenAIChatClient(
    #     api_key="ollama",  # Just a placeholder, Ollama doesn't require API key
    #     base_url=os.getenv("OLLAMA_ENDPOINT"),
    #     model_id=os.getenv("OLLAMA_MODEL"),
    # ).as_agent(
    #     name="WeatherAgent",
    #     instructions="You are a helpful weather agent.",
    #     tools=get_weather,
    # )

    query = "What's the weather like in Seattle?"
    t_query = "What time is it in Seattle? Use a tool call"
    print(f"User: {t_query}")
    result = await time_agent.run(t_query)
    print(f"Agent: {result}\n")


async def streaming_example() -> None:
    """Example of streaming response (get results as they are generated)."""
    print("=== Streaming Response Example ===")

    agent = OpenAIChatClient(
        api_key="ollama",  # Just a placeholder, Ollama doesn't require API key
        base_url=os.getenv("OLLAMA_ENDPOINT"),
        model_id=os.getenv("OLLAMA_MODEL"),
    ).as_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather,
    )

    query = "What's the weather like in Portland?"
    print(f"User: {query}")
    print("Agent: ", end="", flush=True)

    async for update in agent.run_stream(query):
        if update.text:
            print(update.text, end="", flush=True)
    print()  # New line after streaming is complete
    print("\n")


async def main() -> None:
    print("=== Ollama with OpenAI Chat Client Agent Example ===")

    await non_streaming_example()
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())