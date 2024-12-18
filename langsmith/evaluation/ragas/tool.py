from typing import Optional, Type, Dict
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
# from langchain.schema.output_parser import StrOutputParser # 上記とどっちでもいい?

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

from langchain_openai import ChatOpenAI
from retriever import MelosRetriever

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
    ) -> Dict:
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

        get_context_chain = self.retriever | runnable_format_docs
        context = get_context_chain.invoke(input=query, config={"run_name": "コンテキスト抽出"}) # contextはstr

        rag_chain = (
            prompt
            | llm.with_config(config=RunnableConfig(run_name="LLM"))
            | StrOutputParser()
        )

        answer = rag_chain.invoke(input={"question": query, "context": context}, config={"run_name": "RAG"})

        # invoke()したときに最終回答とコンテキストを渡さないと、テストを実行できない。
        # 場合によってはインタフェースを変える、あるいはテスト用に別途作成する必要がある点に注意。
        # 例えば、answerを文字列として返すだけにしていた、など。
        return {"answer": answer, "contexts": context}

# ====== Tool ======
class MelosQATool_SingleOutput(BaseTool):
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
    ) -> Dict:
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