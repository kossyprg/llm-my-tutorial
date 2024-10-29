# Ref https://python.langchain.com/docs/how_to/add_scores_retriever/

from dotenv import load_dotenv
load_dotenv()

from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import chain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict
from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# ==========================================================================
# vectorstoreにデータを格納
# ==========================================================================
def create_vector_store():
    docs = [
        Document(
            page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
            metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
        ),
        Document(
            page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
            metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
        ),
        Document(
            page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
            metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
        ),
        Document(
            page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
            metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
        ),
        Document(
            page_content="Toys come alive and have a blast doing so",
            metadata={"year": 1995, "genre": "animated"},
        ),
        Document(
            page_content="Three men walk into the Zone, three men walk out of the Zone",
            metadata={
                "year": 1979,
                "director": "Andrei Tarkovsky",
                "genre": "thriller",
                "rating": 9.9,
            },
        ),
    ]

    # 公式ドキュメントではPineconeVectorStoreを使用していたが、ここではChromaを使用する
    vectorstore = Chroma.from_documents(
        documents=docs, 
        collection_name="sample",
        embedding=OpenAIEmbeddings()
    )
    return vectorstore

# ==========================================================================
# 類似度スコアをDocumentのmetadata["score"]に格納して返すretrieverを作成する
# ==========================================================================
def main():
    vectorstore = create_vector_store()

    # chainデコレータでRunnableにする
    @chain
    def retriever(query: str) -> List[Document]:
        # zipについてはzip_example()を参照
        # docs: Tuple[Document], scores: Tuple[Float]
        docs, scores = zip(*vectorstore.similarity_search_with_score(query)) 

        # docのmetadataにscoreキーを追加して、そこに類似度スコアを格納
        for doc, score in zip(docs, scores):
            doc.metadata["score"] = score

        return docs

    print(f"[DEBUG] type(retriever): {type(retriever)}") # langchain_core.runnables.base.RunnableLambda
    result = retriever.invoke("dinosaur")
    print(f"result: {result}")
"""
 (Document(metadata={'genre': 'science fiction', 'rating': 7.7, 'year': 1993, 'score': 0.3123563528060913}, page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose'), 
 Document(metadata={'genre': 'animated', 'year': 1995, 'score': 0.4161606729030609}, page_content='Toys come alive and have a blast doing so'), 
 Document(metadata={'director': 'Andrei Tarkovsky', 'genre': 'thriller', 'rating': 9.9, 'year': 1979, 'score': 0.4970167875289917}, page_content='Three men walk into the Zone, three men walk out of the Zone'), 
 Document(metadata={'director': 'Satoshi Kon', 'rating': 8.6, 'year': 2006, 'score': 0.5054081678390503}, page_content='A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea'))
"""

# ==========================================================================
# zip()の動作確認用
# ==========================================================================
def zip_example():
    # zipについて
    # ドキュメントとスコアのペアを含むリストを作成
    documents_with_scores = [
        (Document(metadata={'rating': 'X'}, page_content='a'), 1.1),
        (Document(metadata={'rating': 'Y'}, page_content='b'), 2.2),
        (Document(metadata={'rating': 'Z'}, page_content='c'), 3.3),
    ]

    # *はアンパック演算子でリストをばらす
    print(*documents_with_scores)
    # (Document(metadata={'rating': 'X'}, page_content='a'), 1) (Document(metadata={'rating': 'Y'}, page_content='b'), 1) (Document(metadata={'rating': 'Z'}, page_content='c'), 1)

    # zipを使ってドキュメントとスコアをそれぞれ分離する
    docs, scores = zip(*documents_with_scores)

    print("Docs:", docs)     
    # Docs: (Document(metadata={'rating': 'X'}, page_content='a'), 
    #        Document(metadata={'rating': 'Y'}, page_content='b'), 
    #        Document(metadata={'rating': 'Z'}, page_content='c'))

    print("Scores:", scores)
    # Scores: (1.1, 2.2, 3.3)

    # docのmetadataにscoreキーを追加して、そこにスコアを格納
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score
    
    print("Docs:", docs)
    # Docs: (Document(metadata={'rating': 'X', 'score': 1.1}, page_content='a'), 
    #        Document(metadata={'rating': 'Y', 'score': 2.2}, page_content='b'), 
    #        Document(metadata={'rating': 'Z', 'score': 3.3}, page_content='c'))

# ==========================================================================
# SelfQueryRetrieverを用いる場合。
# 100%うまくいく訳ではなく、検証時の成功率は3回中1回だった。
# SelfQueryRetrieverについては下記を参照。
# Ref https://python.langchain.com/docs/how_to/self_query/
# ==========================================================================
def main_self_query_retriever():
    metadata_field_info = [
        AttributeInfo(
            name="genre",
            description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year the movie was released",
            type="integer",
        ),
        AttributeInfo(
            name="director",
            description="The name of the movie director",
            type="string",
        ),
        AttributeInfo(
            name="rating", description="A 1-10 rating for the movie", type="float"
        ),
    ]
    document_content_description = "Brief summary of a movie"
    llm = ChatOpenAI(name="gpt-4o", temperature=0)

    class CustomSelfQueryRetriever(SelfQueryRetriever):
        def _get_docs_with_query(
            self, query: str, search_kwargs: Dict[str, Any]
        ) -> List[Document]:
            """Get docs, adding score information."""
            docs, scores = zip(
                *self.vectorstore.similarity_search_with_score(query, **search_kwargs)
            )
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            return docs
    
    vectorstore = create_vector_store()

    retriever = CustomSelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
    )

    result = retriever.invoke("dinosaur movie with rating less than 8")
    print(result)
    # 内部で以下のクエリが作成される
    """
    {
        "query": "dinosaur",
        "filter": "and(eq(\"genre\", \"science fiction\"), lt(\"rating\", 8))"
    }
    """

    # 出力
    """
    (Document(metadata={'genre': 'science fiction', 'rating': 7.7, 'year': 1993, 'score': 0.3123563528060913}, 
    page_content='A bunch of scientists bring back dinosaurs and mayhem breaks loose'),)
    """

    # 失敗するときはクエリが不適で何も引っかからない
    """
    {
        "query": "dinosaur",
        "filter": "and(eq(\"rating\", 8))" # 該当なし
    }
    """

# ==========================================================================
# 1つのDocumentに複数のベクトルを関連付けたいときに役立つMultiVectorRetrieverを
# 使う場合に、類似度スコアを付与する方法
# ==========================================================================
def main_multi_vector_retriever():
    # The storage layer for the parent documents
    docstore = InMemoryStore()
    fake_whole_documents = [
        ("fake_id_1", Document(page_content="fake whole document 1")),
        ("fake_id_2", Document(page_content="fake whole document 2")),
    ]
    docstore.mset(fake_whole_documents)

    # sub-documentを作成
    docs = [
        Document(
            page_content="A snippet from a larger document discussing cats.",
            metadata={"doc_id": "fake_id_1"},
        ),
        Document(
            page_content="A snippet from a larger document discussing discourse.",
            metadata={"doc_id": "fake_id_1"},
        ),
        Document(
            page_content="A snippet from a larger document discussing chocolate.",
            metadata={"doc_id": "fake_id_2"},
        ),
    ]

    vectorstore = Chroma.from_documents(
        documents=docs, 
        collection_name="sample",
        embedding=OpenAIEmbeddings()
    )

    # MultiVectorRetriever は BaseRetriever を継承している
    class CustomMultiVectorRetriever(MultiVectorRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            """Get documents relevant to a query.
            Args:
                query: String to find relevant documents for
                run_manager: The callbacks handler to use
            Returns:
                List of relevant documents
            """
            results = self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )

            # Map doc_ids to list of sub-documents, adding scores to metadata
            id_to_doc = defaultdict(list)
            for doc, score in results:
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    doc.metadata["score"] = score
                    id_to_doc[doc_id].append(doc)

            # Fetch documents corresponding to doc_ids, retaining sub_docs in metadata
            docs = []
            for _id, sub_docs in id_to_doc.items():
                docstore_docs = self.docstore.mget([_id])
                if docstore_docs:
                    if doc := docstore_docs[0]:
                        doc.metadata["sub_docs"] = sub_docs
                        docs.append(doc)

            return docs
    
    retriever = CustomMultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        search_kwargs={"k": 1} # defaultが4なので3以下に減らす
    )

    # サブドキュメントに"cat"が含まれる親ドキュメントを取得できる
    res = retriever.invoke("cat")
    print(res)
    """
    [Document(metadata={'sub_docs': [Document(metadata={'doc_id': 'fake_id_1', 'score': 0.3372919261455536}, page_content='A snippet from a larger document discussing cats.')]}, page_content='fake whole document 1')]
    """

if __name__ == "__main__":
    # 類似度スコアをDocumentのmetadata["score"]に格納して返すretrieverを作成する
    main()

    # zip()の動作確認のためのスクリプト
    # zip_example()

    # SelfQueryRetrieverを用いる場合
    main_self_query_retriever()

    # MultiVectorRetrieverを用いる場合
    main_multi_vector_retriever()
