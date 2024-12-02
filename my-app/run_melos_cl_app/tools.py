import chainlit as cl

from typing import Optional, Type, List, Any
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

from typing import Any, Dict, List, Optional, Set
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig

from langchain_core.runnables.utils import (
    Input,
    Output
)

from collections.abc import (
    AsyncIterator,
)

from retrievers import MelosRetriever

# ====== Toolのargs_schema ======
class RetrieveToolInput(BaseModel):
    query: str = Field(description="user query")

class MelosQATool(BaseTool):
    name: str = "走れメロス質疑応答ツール"
    description: str = "走れメロスに関する質問に答える必要があるときに有用"
    args_schema: Type[BaseModel] = RetrieveToolInput
    return_direct: bool = False
    retriever: MelosRetriever
    
    def _get_rag_chain(self):
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

        return rag_chain
        
    # =============== sync version ===============
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        chain = self._get_rag_chain()
        return chain.invoke(query, config=RunnableConfig(run_name="RAG"))
    
    # =============== async version ===============
    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        callbacks: List[Any] = None
    ) -> str:
        chain = self._get_rag_chain()
        return await chain.ainvoke(query, config=RunnableConfig(run_name="RAG"))

    # =============== (async) streaming ===============
    async def astream(
        self,
        query: str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """
        Default implementation of astream, which calls ainvoke.
        Subclasses should override this method if they support streaming output.

        Args:
            input: The input to the Runnable.
            config: The config to use for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the Runnable.
        """
        # 注意：この実装だと on_tool 系のコールバックが実行されない
        #
        # 実装メモ
        # MelosQATool --|> BaseTool --|> RunnableSerializable --|> Runnable
        # Runnable に astream の定義があるが ainvoke を呼んでいるだけ。
        # 継承先で実装しないといけないが、BaseToolではオーバーライドしていないので想定していなさそう？
        # コールバックの処理をしていないので、on_tool_start とかは呼ばれないことに注意。
        chain = self._get_rag_chain()
        async for chunk in chain.astream(
            query, 
            config=config
        ):
            yield chunk # str