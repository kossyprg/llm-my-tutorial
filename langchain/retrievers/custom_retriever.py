# Ref https://python.langchain.com/docs/how_to/custom_retriever/

from dotenv import load_dotenv
load_dotenv()

# 独自のRetrieverを作成するには...
# - BaseRetrieverクラスを継承 > 自動的にRunnableとして扱われて都合がよい
# - _get_relevant_documents(必須) を実装
# - _aget_relevant_documents(optional) を実装
#
# RunnableLambda または RunnableGenerator を使うこともできるが、
# on_retriever_startの代わりにon_chain_startが走るなどの違いがある

from typing import List
import asyncio
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

# テスト用のドキュメント
documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"type": "dog", "trait": "loyalty"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"type": "cat", "trait": "independence"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"type": "fish", "trait": "low maintenance"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"type": "bird", "trait": "intelligence"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"type": "rabbit", "trait": "social"},
    ),
]

# BaseRetrieverはpydanticの BaseModel を継承しているので
# 最初に引数を宣言すればその時点でコンストラクタが完成。
# __init__()は書かない方がよさそう。
# Ref https://docs.pydantic.dev/2.8/concepts/models/
class ToyRetriever(BaseRetriever):
    """ ユーザのクエリを含む k 件のチャンクを返す
    
    これは _get_relevant_documents のみを実装している。

    もしリトリーバーがファイルアクセスやネットワークアクセスを含む場合、
    非同期実装である_aget_relevant_documentsを使用することでパフォーマンスの向上が期待できる。

    通常、Runnableの場合、非同期のデフォルト実装が提供されており、
    それは同期実装を別のスレッドで実行する形で委譲される。
    """

    documents: List[Document]
    """List of documents to retrieve from."""
    k: int
    """Number of top results to return"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        matching_documents = []
        for document in documents:
            if len(matching_documents) > self.k:
                return matching_documents

            if query.lower() in document.page_content.lower():
                matching_documents.append(document)
        return matching_documents

    # Optional: Provide a more efficient native implementation by overriding
    # _aget_relevant_documents
    # async def _aget_relevant_documents(
    #     self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    # ) -> List[Document]:
    #     """Asynchronously get documents relevant to a query.

    #     Args:
    #         query: String to find relevant documents for
    #         run_manager: The callbacks handler to use

    #     Returns:
    #         List of relevant documents
    #     """

def main_sync():
    retriever = ToyRetriever(documents=documents, k=3)
    res = retriever.invoke("that")
    print(f"[main_sync invoke] res: {res}")
    # [main_sync invoke] res: 
    # [Document(
    #   metadata={'type': 'cat', 'trait': 'independence'}, 
    #   page_content='Cats are independent pets that often enjoy their own space.'
    # ), 
    # Document(
    #   metadata={'type': 'rabbit', 'trait': 'social'}, 
    #   page_content='Rabbits are social animals that need plenty of space to hop around.'
    # )]

    # batch
    res = retriever.batch(["dog", "cat"])
    print(f"[main_sync batch] res: {res}")
    # [main_sync batch] res: 
    # [
    #   [Document(
    #       metadata={'type': 'dog', 'trait': 'loyalty'}, 
    #       page_content='Dogs are great companions, known for their loyalty and friendliness.'
    #   ), 
    #   Document(
    #       metadata={'type': 'cat', 'trait': 'independence'}, 
    #       page_content='Cats are independent pets that often enjoy their own space.'
    #   )]
    # ]

async def main_async():
    retriever = ToyRetriever(documents=documents, k=3)

    # runnableなのでainvokeも呼び出せるよ
    res = await retriever.ainvoke("that")
    print(f"[main_async] res: {res}")
    # [main_async] res: 
    # [Document(
    #   metadata={'type': 'cat', 'trait': 'independence'}, 
    #   page_content='Cats are independent pets that often enjoy their own space.'
    #  ), 
    #  Document(
    #   metadata={'type': 'rabbit', 'trait': 'social'}, 
    #   page_content='Rabbits are social animals that need plenty of space to hop around.'
    # )]

    # 登録されているイベントの確認
    async for event in retriever.astream_events("bar", version="v1"):
        print(event)
    # {'event': 'on_retriever_start', 'run_id': '(omitted)', 'name': 'ToyRetriever', 'tags': [], 'metadata': {}, 'data': {'input': 'bar'}, 'parent_ids': []}
    # {'event': 'on_retriever_stream', 'run_id': '(omitted)', 'tags': [], 'metadata': {}, 'name': 'ToyRetriever', 'data': {'chunk': []}, 'parent_ids': []}
    # {'event': 'on_retriever_end', 'name': 'ToyRetriever', 'run_id': '(omitted)', 'tags': [], 'metadata': {}, 'data': {'output': []}, 'parent_ids': []}

if __name__ == "__main__":
    main_sync()
    asyncio.run(main_async())

