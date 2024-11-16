## 走れメロス検索プログラム

「走れメロス」の検索を行うツールです。

### 技術要素
- RAGの基本構成
- `BaseRetriever` を継承した `Retriever` の実装
- `BaseTool` を継承した `Tool` の実装
- LangSmith上に表示される`Run`の名前を変更する

### トレースログ

![langsmith log](img/langsmith_log.png)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"
HF_TOKEN="<your-hf-token>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="run_melos_program"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t run_melos_program .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
docker run -it --rm -v "%cd%":/home/user/app run_melos_program /bin/bash
```

4. スクリプトを実行してください。

```bash
python main.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

