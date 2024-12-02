from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_community.document_loaders import TextLoader

def setup_vectorstore(embeddings):
    vectorstore = Chroma(
        collection_name="melos_collection",
        embedding_function=embeddings,
        persist_directory="./melos_db",  # Where to save data locally, remove if not necessary
    )
    vectorstore.reset_collection() # add_documents()を繰り返すと同じチャンクが重複して入ってしまうのでリセットする

    loader = TextLoader("./run_melos.txt", encoding="utf-8")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["。"], # 改行がないので句点にする
        chunk_size=150,    # チャンクの文字数
        chunk_overlap=0,   # チャンクオーバーラップの文字数
    )

    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    # チャンクの先頭に句点が来てしまうのでそれを削除する
    new_docs = [Document(page_content=d.page_content.lstrip("。"), metadata=d.metadata) for d in docs]

    # 適当なメタデータを付与してみるだけ
    for d in new_docs:
        d.metadata["author"] = "太宰 治"
        d.metadata["link"] = "https://www.aozora.gr.jp/cards/000035/files/1567_14913.html"
        
    vectorstore.add_documents(documents=new_docs)
    return vectorstore