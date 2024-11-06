from dotenv import load_dotenv
load_dotenv()
# import chainlit as cl
import requests
from bs4 import BeautifulSoup
from typing import Optional, Type, List, Any
from pydantic import BaseModel, Field

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
# from langchain.schema.output_parser import StrOutputParser # 上記とどっちでもいい?

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

# 青空文庫の特定のページから文書をスクレイピングする
# 「走れメロス」以外は動作確認していない
def scrape_aozora_page(url):
    response = requests.get(url)

    # エンコーディングを自動検出して設定。SHIFT_JISのはず。
    response.encoding = response.apparent_encoding  

    # HTMLを解析
    soup = BeautifulSoup(response.text, 'html.parser')

    # 指定のdivを取得
    main_text_div = soup.find('div', class_='main_text')

    # なければエラー
    if main_text_div is None:
        print("[ERROR] Cannot find any document.")
        return None

    # 送り仮名が振ってある漢字を処理
    for ruby_tag in main_text_div.find_all('ruby'):
        rb_text = ruby_tag.find('rb').get_text(strip=True) if ruby_tag.find('rb') else "" # 漢字（例：邪知暴虐）
        rt_text = ruby_tag.find('rt').get_text(strip=True) if ruby_tag.find('rt') else "" # 送り仮名（例：じゃちぼうぎゃく）
        ruby_tag.replace_with(f"{rb_text}({rt_text})")  # <ruby>全体を置き換え
    
    # テキストのみ取得
    main_text = main_text_div.get_text(separator='', strip=True)
    return main_text

def save_melos_text_from_aozora_page():
    main_text = scrape_aozora_page("https://www.aozora.gr.jp/cards/000035/files/1567_14913.html")
    with open("run_melos.txt", "w", encoding="utf-8") as file:
        file.write(main_text)

# ====== Retriever ======
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

# ====== Toolのargs_schema ======
class RetrieveToolInput(BaseModel):
    query: str = Field(description="user query")

# ====== Tool ======
class MelosQATool(BaseTool):
    name: str = "走れメロス質疑応答ツール"
    description: str = "走れメロスに関する質問に答える必要があるときに有用"
    args_schema: Type[BaseModel] = RetrieveToolInput
    return_direct: bool = False
    retriever: MelosRetriever = None

    # RAGを実行するツール
    # Ref https://python.langchain.com/docs/tutorials/rag/
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        # 参照情報を連結する関数
        def format_docs(docs):
            context = ""
            for d in docs:
                context += f"{d.page_content}\n"
            return context
        
        # RunnableLambdaを使うことでメソッドをRunnableにできる
        # Ref https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableLambda.html
        # with_configでconfigを指定することでLangSmithで表示されるRunの名前を変更できる
        # Ref https://docs.smith.langchain.com/old/cookbook/tracing-examples/runnable-naming#example-2-runnable-lambda
        runnable_format_docs = RunnableLambda(func=format_docs).with_config(config=RunnableConfig(run_name="参照情報整形メソッド"))

        prompt = PromptTemplate.from_template("参照情報を参考にして、質問に答えなさい。\n\n" + \
                                              "## 質問\n" + \
                                              "{question}\n\n" + \
                                              "## 参照情報\n" + \
                                              "{context}")
        
        llm = ChatOpenAI(model="gpt-4o")

        get_context_chain = (self.retriever | runnable_format_docs).with_config(config=RunnableConfig(run_name="コンテキスト抽出"))
        rag_chain = (
            {"context": get_context_chain, "question": RunnablePassthrough()}
            | prompt
            | llm.with_config(config=RunnableConfig(run_name="LLM"))
            | StrOutputParser()
        )

        return rag_chain.invoke(query, config=RunnableConfig(run_name="RAG"))

if __name__ == "__main__":
    # 走れメロスを青空文庫から引っ張ってきてtxtファイルとして保存
    if False:
        save_melos_text_from_aozora_page()

    # Embeddingsの定義
    # Ref https://api.python.langchain.com/en/latest/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # 50言語に対応, 384次元
    # model_name = "intfloat/multilingual-e5-large"          # 94言語に対応, 1024次元.
    # model_name = "sentence-transformers/all-mpnet-base-v2" # 英語しか対応してないので難しいはず
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./cache_embed_model",
    )

    vectorstore = Chroma(
        collection_name="melos_collection",
        embedding_function=embeddings,
        persist_directory="./melos_db",  # Where to save data locally, remove if not necessary
    )

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

    vectorstore.add_documents(documents=new_docs)
    retriever = MelosRetriever(vectorstore=vectorstore, k=4, name="走れメロス検索エンジン")
    tool = MelosQATool(retriever=retriever)

    # ツール実行
    result = tool.invoke("メロスは太陽の沈む速度の何倍で走りましたか？")
    print(f"result: {result}")

    # 何度も実行すると同じチャンクが重複して入ってしまうので削除する
    vectorstore.delete_collection()
