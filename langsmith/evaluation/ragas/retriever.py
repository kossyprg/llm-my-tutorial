
from typing import Optional, Type, List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

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