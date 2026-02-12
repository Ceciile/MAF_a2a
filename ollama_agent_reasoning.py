"""
诊断脚本：检测 Microsoft Agent Framework + Ollama 的问题

运行这个脚本可以帮助你：
1. 检查 Ollama 是否运行
2. 检查依赖版本
3. 测试 API 调用
4. 找出你的代码问题所在
"""
# Copyright (c) Microsoft. All rights reserved.

import asyncio
from datetime import datetime
import os

from agent_framework import tool
from agent_framework.ollama import OllamaChatClient
from agent_framework.openai import OpenAIChatClient

@tool(approval_mode="never_require")
def get_time():
    """Get the current time."""
    return f"The current time is {datetime.now().strftime('%I:%M %p')}."

"""
Ollama Agent Reasoning Example

This sample demonstrates implementing a Ollama agent with reasoning.

Ensure to install Ollama and have a model running locally before running the sample
Not all Models support reasoning, to test reasoning try qwen3:8b
Set the model to use via the OLLAMA_MODEL_ID environment variable or modify the code below.
https://ollama.com/

"""


async def main() -> None:
    print("=== Response Reasoning Example ===")
    # agent = OllamaChatClient().as_agent(
    #     name="TimeAgent",
    #     instructions="You are a helpful agent answer in one sentence.",
    #     default_options={"think": True},  # Enable Reasoning on agent level
    # )
    # query = "Hey what is 3+4? Can you explain how you got to that answer?"
    # print(f"User: {query}")
    # # Enable Reasoning on per request level
    # result = await agent.run(query)
    # reasoning = "".join((c.text or "") for c in result.messages[-1].contents if c.type == "text_reasoning")
    # print(f"Reasoning: {reasoning}")
    # print(f"Answer: {result}\n")

    client = OpenAIChatClient(
        api_key="ollama",  # Just a placeholder, Ollama doesn't require API key
        base_url=os.getenv("OLLAMA_ENDPOINT"),
        model_id=os.getenv("OLLAMA_MODEL"),
    )

    message = "What time is it? Use a tool call"
    stream = True
    print(f"User: {message}")
    if stream:
        print("Assistant: ", end="")
        async for chunk in client.get_streaming_response(message, tools=get_time, stream=True):
            if str(chunk):
                print(str(chunk), end="")
        print("")
    else:
        response = await client.get_response(message, tools=get_time)
        print(f"Assistant: {response}")




if __name__ == "__main__":
    asyncio.run(main())