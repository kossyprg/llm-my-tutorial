from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

def setup_embeddings():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # 50言語に対応, 384次元
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./cache_embed_model",
    )
    return embeddings

def setup_vectorstore(embeddings):
    vectorstore = Chroma(
        collection_name="melos_collection",
        embedding_function=embeddings,
        persist_directory="./melos_db",  # Where to save data locally, remove if not necessary
    )

    return vectorstore

# （デバッグ用）ドキュメントをコンソールに表示する
def print_docs(docs, label="chunk"):
    for i, doc in enumerate(docs):
        print("="*5 + f"{label} {i+1}" + "="*5)
        print(doc.page_content)
        print(doc.metadata)

if __name__ == "__main__":
    print("main")
    embeddings  = setup_embeddings()
    vectorstore = setup_vectorstore(embeddings)

    loader = TextLoader("./run_melos.txt", encoding="utf-8")
    documents = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["。"],
        chunk_size=500,
        chunk_overlap=0,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        separators=["。"], 
        chunk_size=15,
        chunk_overlap=0,
    )

    # （デバッグ用）親チャンクと子チャンクの数を調べるだけ
    parent_docs = parent_splitter.split_documents(documents)
    child_docs  = child_splitter.split_documents(documents)
    print(f"[DEBUG] len(parent_docs): {len(parent_docs)}") # 22
    print(f"[DEBUG] len(child_docs): {len(child_docs)}")   # 449

    store = InMemoryStore()

    # k は子チャンクの検索個数を表し、k = 子チャンクの検索個数 >= 親チャンクの検索個数である。
    # 子チャンクの検索個数 > 親チャンクの検索個数 となるのは、
    # ヒットした子チャンクの属する親チャンクが重複した場合。
    k = 6
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
        search_kwargs={"k": k}
    )

    retriever.add_documents(documents=documents)

    # 親チャンクのキーと内容を確認する
    for i, key in enumerate(list(store.yield_keys())):
        print(f"num: {i+1}")
        print(f"key: {key}")
        parent_doc = store.mget(keys=[key]) # List[Document] で要素は1個だけ
        print(f"parent_doc[0].page_content: {parent_doc[0].page_content}\n")

    query = "セリヌンティウス"

    # （デバッグ用）子チャンクの検索結果
    sub_docs = retriever.vectorstore.similarity_search(query, k = k)
    print_docs(sub_docs, label="child chunk")

    # 子チャンクでベクトル検索をかけて、
    # ヒットした子チャンクの属する親チャンクを取得する
    retrieved_docs = retriever.invoke(query)
    print_docs(retrieved_docs, label="parent chunk")

    # 何度も実行すると同じチャンクが重複して入ってしまうので削除する
    vectorstore.delete_collection()
