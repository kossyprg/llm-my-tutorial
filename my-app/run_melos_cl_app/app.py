from dotenv import load_dotenv
load_dotenv()
import chainlit as cl

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

from callbacks import BaseCustomCallbackHandler
from tools import MelosQATool
from retrievers import MelosRetriever

@cl.on_chat_start
async def handle_chat_start():
    # Embeddingsの定義
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # 50言語に対応, 384次元
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./cache_embed_model",
    )

    collection_name = "melos_collection"
    vectorstore = Chroma(
        collection_name="melos_collection",
        embedding_function=embeddings,
        persist_directory="./melos_db",  # Where to save data locally, remove if not necessary
    )

    collection = vectorstore._client.get_collection(name=collection_name)

    if collection.count() == 0:
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
    retriever = MelosRetriever(vectorstore=vectorstore, name="走れメロス検索エンジン")
    tool = MelosQATool(retriever=retriever)
    cl.user_session.set("tool", tool)

@cl.on_message
async def handle_on_message(message: cl.Message):    
    msg_out = cl.Message(content="")
    tool = cl.user_session.get("tool")

    # async でツールを実行
    result = await tool.ainvoke(
        input=message.content, 
        config=RunnableConfig(
            callbacks=[BaseCustomCallbackHandler(msg_out), cl.AsyncLangchainCallbackHandler()]
        )
    )

    msg_out.content = result

    # display="side"の場合、cl.Textのname属性がメッセージの中にリンクとして入っていないと表示されないので注意
    # Ref https://github.com/Chainlit/chainlit/issues/729
    source_names = [text_el.name for text_el in msg_out.elements]
    if source_names:
        msg_out.content += f"\n{', '.join(source_names)}"
    else:
        msg_out.content += "\n参照情報が見つかりませんでした"

    await msg_out.send()
    