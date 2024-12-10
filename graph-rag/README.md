## Graph RAG

Ref: https://ja.wikipedia.org/wiki/サザエさん#主な登場人物・ペット

### 技術要素

- Langgraph の基本構成
- chainlit における streaming の実装

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="graph-rag"

NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=<your-neo4j-username>       # e.g neo4j
NEO4J_PASSWORD=<your-neo4j-password>
AURA_INSTANCEID=<your-aura-instanceid>
AURA_INSTANCENAME=<your-aura-instancename> # e.g Instance01
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t graph-rag .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```bash
docker run -it --rm -v "%cd%":/home/user/app --name graph-rag graph-rag /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python main.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### Graph RAG

[graph_rag_sazae.py](graph_rag_sazae.py)

参考：[GraphRAGをわかりやすく解説](https://qiita.com/ksonoda/items/98a6607f31d0bbb237ef)
（参照日：2024/12/10）

文書からNeo4jへグラフデータベースを登録、ユーザの質問文からCypherクエリを生成してRAGに使う流れを確認しています。