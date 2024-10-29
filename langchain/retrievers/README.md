## Retrievers チュートリアル

Retrievers に関するチュートリアルを実行するためのソースファイル群です。

参考：[langchain Retrievers](https://python.langchain.com/docs/how_to/#retrievers)

## 実行方法

1. `.env` ファイルを作成して環境変数を記述してください。

```
OPENAI_API_KEY="<your-openai-api-key>"

# Langsmithでトレースする場合は以下4つが必要
# LANGCHAIN_PROJECTは任意の名前を設定できる
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="<your-langsmith-api-key>"
LANGCHAIN_PROJECT="retrievers-tutorial"
```

2. `Dockerfile` を使用してビルドします。

```bash
docker build -t retrievers .
```

3. ビルドしたイメージを実行してください。`-v`オプションでボリュームをマウントすると、ソースコードの修正がコンテナ環境にも反映されます。

Windows(cmd)の場合
```cmd
REM For Windows(cmd)
docker run -it --rm -v "%cd%":/home/user/app retrievers /bin/bash
```

4. 所望のスクリプトを実行してください。

```bash
python custom_retriever.py
```

5. 終了する際は`exit`を入力してください

```bash
exit
```

## ソースコード

### Retriever を自作する
[custom_retriever.py](custom_retriever.py)

`BaseRetriever` を継承して `Retriever` を自作する方法です。
`_get_relevant_documents()` は必須です。

```python
from langchain_core.retrievers import BaseRetriever

class ToyRetriever(BaseRetriever):
    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 種々の処理
        # メンバ変数は self.k のようにアクセスする
        return matching_documents
```
### 類似度スコアを付与する方法
[add_scores_retriever.py](add_scores_retriever.py)

`Document` のメタデータに類似度スコアが付与されるようにする方法です。
`similarity_search_with_score()` を使うことで対応できます。

戻り値は `List[Tuple[Document, float]]` なので、`zip()` などを使ってメタデータとして格納します。

```python
@chain
def retriever(query: str) -> List[Document]:
    # zipについてはzip_example()を参照
    # docs: Tuple[Document], scores: Tuple[Float]
    docs, scores = zip(*vectorstore.similarity_search_with_score(query)) 

    # docのmetadataにscoreキーを追加して、そこに類似度スコアを格納
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs
```

参考：
[How to create a custom Retriever](https://python.langchain.com/docs/how_to/custom_retriever/)
