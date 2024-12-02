from dotenv import load_dotenv
load_dotenv()
import chainlit as cl

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

from embeddings import load_hf_embeddings
from vectorstore import setup_vectorstore
from callbacks import BaseCustomCallbackHandler, ChainlitUICallbackHandler
from tools import MelosQATool
from retrievers import MelosRetriever

@cl.on_chat_start
async def handle_chat_start():
    # Embeddingsの定義
    embeddings = load_hf_embeddings()
    vectorstore = setup_vectorstore(embeddings)
    retriever = MelosRetriever(vectorstore=vectorstore, name="走れメロス検索エンジン")
    tool = MelosQATool(retriever=retriever)
    cl.user_session.set("tool", tool)

@cl.on_message
async def handle_on_message(message: cl.Message):    
    msg_out = cl.Message(content="")
    tool = cl.user_session.get("tool")

    callbacks = [BaseCustomCallbackHandler(msg_out), ChainlitUICallbackHandler()]
    run_name = "RAG"
    
    streaming_mode = False
    if streaming_mode:
        # MelosQATool は BaseTool を継承しており、astream の定義は独自に実装している。
        # それにより、on_tool 系のコールバック関数が発動しない。
        # BaseTool で astream をオーバーライドしていないことを考慮すると、
        # ツールの streaming は想定していないと思われる。
        # ここではやや強引に実装した。
        async for chunk in tool.astream(
            query=message.content,
            config={"callbacks": callbacks, "run_name": run_name}
        ):
            await msg_out.stream_token(chunk)
    else:
        msg_out.content = await tool.ainvoke(
            input=message.content, 
            config={"callbacks": callbacks}
        )

    # display="side"の場合、cl.Textのname属性がメッセージの中にリンクとして入っていないと表示されないので注意
    # Ref https://github.com/Chainlit/chainlit/issues/729
    source_names = [text_el.name for text_el in msg_out.elements]
    if source_names:
        msg_out.content += f"\n{', '.join(source_names)}"
    else:
        msg_out.content += "\n参照情報が見つかりませんでした"

    await msg_out.send()
    