from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import Any, List

# retrieverの実装方法は以下
# Ref https://python.langchain.com/docs/how_to/custom_retriever/
class MelosRetriever(BaseRetriever):
    vectorstore: Any
    k: int = 4

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=self.k)

    # langchain_chroma の asimilarity_search メソッドについては以下
    # Ref https://api.python.langchain.com/en/latest/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.asimilarity_search
    async def _aget_relevant_documents(
        self, 
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 類似度スコアを付与する
        # Ref https://python.langchain.com/docs/how_to/add_scores_retriever/
        results = await self.vectorstore.asimilarity_search_with_score(query) # List[Tuple[Document, float]]
        docs, scores = zip(*results)
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score
        return docs

        # Document のリストだけを返すなら以下でいい
        # return await self.vectorstore.asimilarity_search(query=query, k=self.k)