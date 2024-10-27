# Ref https://python.langchain.com/docs/how_to/agent_executor/
# AgentExecutorはLegacyで、LangGraphへの移行が推奨されている

from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Tavily ツール
search = TavilySearchResults(max_results=2)
# テスト
# print(f'search invoke: {search.invoke("what is the weather in SF")}')

# retriever ツール
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()
# テスト
# print(f'retriever invoke: {retriever.invoke("how to upload a dataset")[0]}')

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="langsmith_search",
    description="Search for information about LangSmith. For any questions about LangSmith, you must use this tool!",
)

tools = [search, retriever_tool]
model = ChatOpenAI(model="gpt-4o")


def model_with_tools_test():
    model_with_tools = model.bind_tools(tools)   

    # ツールつきのモデルをinvokeする
    response = model_with_tools.invoke([HumanMessage(content="Hi!")])
    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")
    # ContentString: Hello! How can I assist you today?
    # ToolCalls: []

    # ツールが呼ばれそうなクエリを入力
    response = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])
    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")
    # ContentString:
    # ToolCalls: [{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': (omitted), 'type': 'tool_call'}]

def agent_test():
    # エージェントをガイドするためのプロンプト
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # print(prompt.messages)
    # [SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template='You are a helpful assistant'), additional_kwargs={}), 
    # MessagesPlaceholder(variable_name='chat_history', optional=True), 
    # HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], input_types={}, partial_variables={}, template='{input}'), additional_kwargs={}), 
    # MessagesPlaceholder(variable_name='agent_scratchpad')]

    # モデル、ツール、プロンプトを渡してAgentを作成する
    # model_with_toolsではないことに注意。
    # これは create_tool_calling_agent() が内部で.bind_tools()を実行するため。
    agent = create_tool_calling_agent(model, tools, prompt)

    # 頭脳にあたるagentとツールをAgentExecutorに渡す
    # AgentExecutorはagentを繰り返し呼び出し、toolを実行する。
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # ツールを呼び出す必要のないクエリ
    # finish_reason: stop で終了
    output = agent_executor.invoke({"input": "hi!"})
    print(f"output: {output}")

    # tool_callsが実行される
    output = agent_executor.invoke({"input": "whats the weather in sf?"})
    print(f"output: {output}")

def agent_with_chat_history_test():
    # 以下はagent_testと同じ
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # 明示的にchat_historyを書くことができる。
    # キー名を"chat_history"とするのはプロンプトがそのようになっているから。
    output = agent_executor.invoke(
        {
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
            "input": "what's my name?",
        }
    )

    print(f"output: {output}") # your name is bob!

def agent_with_memory_test():
    # 以下はagent_testと同じ
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # 自動的に履歴を保存する
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            # 新たにChatMessageHistoryを作成
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history, # エージェントにメッセージ履歴を提供する
        input_messages_key="input", # クエリとして送信されるメッセージを扱うためのキー
        history_messages_key="chat_history", # 保存された履歴を基に応答を行う際に参照されるキー
    )

    output = agent_with_chat_history.invoke(
        {"input": "hi! I'm kossy!"},
        config={"configurable": {"session_id": "<foo>"}},
    )
    print(f"output: {output}")
    # output: {'input': "hi! I'm kossy!", 'chat_history': [], 'output': 'Hello Kossy! How can I assist you today?'}

    output = agent_with_chat_history.invoke(
        {"input": "what's my name?"},
        config={"configurable": {"session_id": "<foo>"}},
    )
    print(f"output: {output}")
    # output: {'input': "what's my name?", 'chat_history': [HumanMessage(content="hi! I'm kossy!", additional_kwargs={}, response_metadata={}), AIMessage(content='Hello Kossy! How can I assist you today?', additional_kwargs={}, response_metadata={})], 'output': 'Your name is Kossy! How can I help you today?'}

    print(f"store: {store}")
    # store: {'<foo>': InMemoryChatMessageHistory(messages=[
    # HumanMessage(content="hi! I'm kossy!", additional_kwargs={}, response_metadata={}), 
    # AIMessage(content='Hello Kossy! How can I assist you today?', additional_kwargs={}, response_metadata={}), 
    # HumanMessage(content="what's my name?", additional_kwargs={}, response_metadata={}), 
    # AIMessage(content='Your name is Kossy! How can I help you today?', additional_kwargs={}, response_metadata={})])}


if __name__ == "__main__":
    # .bind_tools()でツールを渡したモデルを使う。
    # ツールを使う指示がとぶだけで、実際にツールは呼ばれない。
    model_with_tools_test()

    # エージェントのテスト。メモリ機能はなし
    agent_test()

    # 会話履歴つきのエージェントのテスト
    agent_with_chat_history_test()

    # メモリ機能付きのエージェントのテスト
    agent_with_memory_test()