

## 技术栈

- **LLM**: Ollama
- **开发环境**: Python >=3.10 + uv

## 开发规范

### Python 依赖管理

poetry 针对除非显式允许（--pre）也可配置开启“允许预发布” allow-prerelease 但会有版本问题：

#### UV 项目依赖 (`pyproject.toml`)

直接使用 pyproject.toml 管理：

```bash
uv python pin python3.14

# 配置
[tool.uv]
prerelease = "allow"

# 添加新依赖
uv add agent-framework --prerelease allow
```

> **Note**: Not all models support all features. Function calling, reasoning, and multimodal capabilities depend on the specific model you're using.

## Recommended Approach

The recommended way to use Ollama with Agent Framework is via the native `OllamaChatClient` from the `agent-framework-ollama` package. This provides full support for Ollama-specific features like reasoning mode.

Alternatively, you can use the `OpenAIChatClient` configured to point to your local Ollama server, which may be useful if you're already familiar with the OpenAI client interface.

## Examples

| File | Description |
|------|-------------|
| [`ollama_with_openai_chat_client.py`](https://github.com/microsoft/agent-framework/blob/main/python/samples/getting_started/agents/ollama/ollama_with_openai_chat_client.py) | Alternative approach using OpenAI Chat Client configured to use local Ollama models. |