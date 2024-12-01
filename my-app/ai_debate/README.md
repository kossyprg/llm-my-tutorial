## AI-debate

AI同士にディベートをさせるアプリケーションです。

### 技術要素

- Langgraph の基本構成
- chainlit における streaming の実装

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"
TAVILY_API_KEY="<your-tavily-api-key>"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="ai-debate"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t ai-debate .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app -p 8000:8000 --name ai-debate ai-debate /bin/bash
```

4. app.py を実行し、[http://localhost:8000/](http://localhost:8000/) にアクセスしてください。

```bash
chainlit run app.py --host 0.0.0.0
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

