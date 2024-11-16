## LangGraph チュートリアル

LangGraph に関するチュートリアルを実行するためのソースファイル群です。

参考：[LangGraph](https://langchain-ai.github.io/langgraph/)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="langgraph-tutorial"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t lg-tut .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app lg-tut /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python simple_agent_example.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### シンプルなAgentの例
[simple_agent_example.py](simple_agent_example.py)

公式ドキュメントの一番最初に掲載されている例です。
点線はconditional edgeと呼ばれ、ツールが必要な場合は`tools` ノードへ、不要なら `END` ノードに向かいます。

![graph diagram of simple agent](img/graph_image.png)

参考：
[LangGraph example](https://langchain-ai.github.io/langgraph/#example)
