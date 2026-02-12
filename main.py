# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
from datetime import datetime
from random import randint, randrange
from typing import TYPE_CHECKING, Annotated, Any

from agent_framework import ChatMessage, tool
from agent_framework.openai import OpenAIChatClient
from agent_framework.ollama import OllamaChatClient

if TYPE_CHECKING:
    from agent_framework import SupportsAgentRun
"""
Ollama with OpenAI Chat Client Example

This sample demonstrates using Ollama models through OpenAI Chat Client by
configuring the base URL to point to your local Ollama server for local AI inference.
Ollama allows you to run large language models locally on your machine.

Environment Variables:
- OLLAMA_ENDPOINT: The base URL for your Ollama server (e.g., "http://localhost:11434/v1/")
- OLLAMA_MODEL: The model name to use (e.g., "mistral", "llama3.2", "phi3")

Demonstration of a tool with approvals.
This sample demonstrates using AI functions with user approval workflows.
It shows how to handle function call approvals without using threads.
"""

conditions = ["sunny", "cloudy", "raining", "snowing", "clear"]

# NOTE: approval_mode="never_require" is for sample brevity. Use "always_require" in production; see samples/getting_started/tools/function_tool_with_approval.py and samples/getting_started/tools/function_tool_with_approval_and_threads.py.
@tool(approval_mode="never_require")
def get_weather(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """Get the weather for a given location."""
    # conditions = ["sunny", "cloudy", "rainy", "stormy"]
    return f"The weather in {location} is {conditions[randint(0, 3)]} with a high of {randint(10, 30)}°C."


# Define a simple weather tool that requires approval
@tool(approval_mode="always_require")
def get_weather_detail(
    location: Annotated[str, "The location to get the weather for."],
) -> str:
    """Get the weather for a given location."""
    return (
        f"The weather in {location} is {conditions[randrange(0, len(conditions))]} and {randrange(-10, 30)}°C, "
        "with a humidity of 66%. "
        f"Tomorrow will be {conditions[randrange(0, len(conditions))]} with a high of {randrange(-10, 30)}°C."
    )


"""
Ensure to install Ollama and have a model running locally before running the sample
Not all Models support function calling, to test function calling try llama3.2 or qwen3:4b
Set the model to use via the OLLAMA_MODEL_ID environment variable or modify the code below.
https://ollama.com/

"""
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



async def handle_approvals_streaming(query: str, agent: "SupportsAgentRun") -> None:
    """Handle function call approvals with streaming responses.

    When we don't have a thread, we need to ensure we include the original query,
    the approval request, and the approval response in each iteration.
    """
    current_input: str | list[Any] = query
    has_user_input_requests = True
    while has_user_input_requests:
        has_user_input_requests = False
        user_input_requests: list[Any] = []

        # Stream the response
        async for chunk in agent.run_stream(current_input, stream=True):
            if chunk.text:
                print(chunk.text, end="", flush=True)

            # Collect user input requests from the stream
            if chunk.user_input_requests:
                user_input_requests.extend(chunk.user_input_requests)

        if user_input_requests:
            has_user_input_requests = True
            # Start with the original query
            new_inputs: list[Any] = [query]

            for user_input_needed in user_input_requests:
                print(
                    f"\n\nUser Input Request for function from {agent.name}:"
                    f"\n  Function: {user_input_needed.function_call.name}"
                    f"\n  Arguments: {user_input_needed.function_call.arguments}"
                )

                # Add the assistant message with the approval request
                new_inputs.append(ChatMessage(role="assistant", contents=[user_input_needed]))

                # Get user approval
                user_approval = await asyncio.to_thread(input, "\nApprove function call? (y/n): ")

                # Add the user's approval response
                new_inputs.append(
                    ChatMessage(
                        role="user",
                        contents=[user_input_needed.to_function_approval_response(user_approval.lower() == "y")],
                    )
                )

            # Update input with all the context for next iteration
            current_input = new_inputs



async def streaming_example() -> None:
    """Example showing AI function with approval requirement."""
    print(f"\n=== Weather Agent with Approval Required ===\n")
    print("=== Streaming Response Example ===")

    async with OpenAIChatClient(
        api_key="ollama",  # Just a placeholder, Ollama doesn't require API key
        base_url=os.getenv("OLLAMA_ENDPOINT"),
        model_id=os.getenv("OLLAMA_MODEL"),
    ).as_agent(
        name="WeatherAgent",
        instructions="You are a helpful weather agent.",
        tools=get_weather_detail,
    ) as agent:
        query = "Can you give me an update of the weather in Portland and detailed weather for Seattle?"
        print(f"User: {query}")

    print(f"\n{agent.name}: ", end="", flush=True)

    await handle_approvals_streaming(query, agent)
    print()  # New line after streaming is complete
    print("\n")




"""
Tool Approvals with Threads

This sample demonstrates using tool approvals with threads.
With threads, you don't need to manually pass previous messages -
the thread stores and retrieves them automatically.
"""


@tool(approval_mode="always_require")
def add_to_calendar(
    event_name: Annotated[str, "Name of the event"], date: Annotated[str, "Date of the event"]
) -> str:
    """Add an event to the calendar (requires approval)."""
    print(f">>> EXECUTING: add_to_calendar(event_name='{event_name}', date='{date}')")
    return f"Added '{event_name}' to calendar on {date}"


async def approval_example() -> None:
    """Example showing approval with threads."""
    print("=== Tool Approval with Thread ===\n")

    async with OpenAIChatClient(
        api_key="ollama",  # Just a placeholder, Ollama doesn't require API key
        base_url=os.getenv("OLLAMA_ENDPOINT"),
        model_id=os.getenv("OLLAMA_MODEL"),
    ).as_agent(
        name="CalendarAgent",
        instructions="You are a helpful calendar assistant.",
        tools=[add_to_calendar],
    ) as agent:
        thread = agent.get_new_thread()
        # Step 1: Agent requests to call the tool
        query = "Add a dentist appointment on March 15th"
        print(f"User: {query}")
    result = await agent.run(query, thread=thread)

    # Check for approval requests
    if result.user_input_requests:
        for request in result.user_input_requests:
            print("\nApproval needed:")
            print(f"  Function: {request.function_call.name}")
            print(f"  Arguments: {request.function_call.arguments}")

            # User approves (in real app, this would be user input)
            approved = True  # Change to False to see rejection
            print(f"  Decision: {'Approved' if approved else 'Rejected'}")

            # Step 2: Send approval response
            approval_response = request.to_function_approval_response(approved=approved)
            result = await agent.run(ChatMessage(role="user",
                        contents=[approval_response]), thread=thread)

    print(f"Agent: {result}\n")


async def rejection_example() -> None:
    """Example showing rejection with threads."""
    print("=== Tool Rejection with Thread ===\n")
    async with OpenAIChatClient(
        api_key="ollama",  # Just a placeholder, Ollama doesn't require API key
        base_url=os.getenv("OLLAMA_ENDPOINT"),
        model_id=os.getenv("OLLAMA_MODEL"),
    ).as_agent(
        name="CalendarAgent",
        instructions="You are a helpful calendar assistant.",
        tools=[add_to_calendar],
    ) as agent:
        thread = agent.get_new_thread()
        query = "Add a team meeting on December 20th"
        print(f"User: {query}")

    result = await agent.run(query, thread=thread)

    if result.user_input_requests:
        for request in result.user_input_requests:
            print("\nApproval needed:")
            print(f"  Function: {request.function_call.name}")
            print(f"  Arguments: {request.function_call.arguments}")

            # User rejects
            print("  Decision: Rejected")

            # Send rejection response
            rejection_response = request.to_function_approval_response(approved=False)
            result = await agent.run(ChatMessage(role="user",
                        contents=[rejection_response]), thread=thread)

    print(f"Agent: {result}\n")





async def main() -> None:
    print("=== Ollama with OpenAI Chat Client Agent Example ===")

    await non_streaming_example()
    # await streaming_example()
    await approval_example()
    await rejection_example()


if __name__ == "__main__":
    asyncio.run(main())