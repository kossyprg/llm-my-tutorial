## Agents チュートリアル

Agentに関するチュートリアルを実行するためのソースファイル群です。

参考：[langchain Agents](https://python.langchain.com/docs/how_to/#agents)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"

# Tavilyを使用する場合は必要
TAVILY_API_KEY="<your-tavily-api-key>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="agent-tutorial"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t agents .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app agents /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python agent_executor.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### 1. AgentExecutor(legacy)
[agent_executor.py](agent_executor.py)

`AgentExecutor` を用いたエージェントの実行例です。公式ドキュメントによると、LangGraphへの移行が推奨されています。

```python
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

参考：
[Build an Agent with AgentExecutor (Legacy)](https://python.langchain.com/docs/how_to/agent_executor/)
