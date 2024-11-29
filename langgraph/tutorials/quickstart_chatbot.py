# Ref https://langchain-ai.github.io/langgraph/tutorials/introduction/
from dotenv import load_dotenv
load_dotenv()
from icecream import ic
from langchain_openai import ChatOpenAI

# シンプルな chatbot の構築
def main_part1():
    from typing import Annotated
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    # チャットボットの構造をステートマシン(有限個の状態を行き来するグラフ)として定義する
    # TypedDict はひとまず辞書と思えばよさそう。キーは messages
    class State(TypedDict):
        # State は messages という1つのキーを持つ辞書。
        # add_messages は reducer function と呼ばれ、state の更新に使われる。
        # reducer function がなければ上書きされる。
        # 今回はリストにメッセージを追加することになる。
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    llm = ChatOpenAI(model="gpt-4o", streaming=True)

    # chatbot は State を入力として受け取って、更新された messages を返す
    # 戻り値は state の末尾に追加される
    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}
    
    # 第一引数: ノード名
    # 第二引数: ノードが使われるときに呼ばれる関数
    graph_builder.add_node("chatbot", chatbot)

    # エントリポイントを定義する。グラフはここからスタートする。
    graph_builder.add_edge(START, "chatbot") # START --> chatbot

    # 終了するノードを定義する。
    graph_builder.add_edge("chatbot", END) # chatbot --> END

    # グラフを実行できるようにするためコンパイルする。
    graph = graph_builder.compile()
    ic(type(graph)) # <class 'langgraph.graph.state.CompiledStateGraph'>

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part1.png")

    user_input = "RAGって何ですか？中学生にもわかるように説明してください。"
    config={"run_name": "(part 1) Basic chatbot"}
    result = graph.invoke({"messages": [("user", user_input)]}, config=config)

    ic(type(result)) # <class 'langgraph.pregel.io.AddableValuesDict'>
    ic(result)

    # streaming を試す
    # Ref https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/#streaming-llm-tokens
    for msg, metadata in graph.stream(
        {"messages": [("user", user_input)]}, 
        config=config, 
        stream_mode="messages"
    ):
        print(msg.content, end="|", flush=True)
    print("\n")

# ツールを追加してシンプルな Agent を実装する
def main_part2():
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    # Ref https://python.langchain.com/docs/integrations/tools/wikipedia/
    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    tools = [tool]
    ic(tool.invoke("HUNTER X HUNTER"))

    from typing import Annotated
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    graph_builder = StateGraph(State)

    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    import json
    from langchain_core.messages import ToolMessage

    # ツールが呼ばれたときにツールを実行する関数を作成する。
    # これは新しいノードにツールを追加することで実現できる。
    # message のなかに tool_calls が含まれていたらツールを実行する BasicToolNode を実装する
    # main_part3() では ToolNode に置き換える。
    class BasicToolNode:
        """A node that runs the tools requested in the last AIMessage."""

        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}
    
    tool_node = BasicToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    # チャットボットの出力に tool_calls があるかを調べ、
    # あれば "tools" ノードに遷移させる router function
    # main_part3() で tools_condition に置き換える
    def route_tools(
        state: State,
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    # The `route_tools` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",   # source
        route_tools, # path
        # 以下の辞書を使用すると、条件の出力を特定のノードとして解釈するようグラフに指示できます。
        # デフォルトでは出力をそのまま（identity function）使用しますが、
        # 「tools」というノード以外の名前を使用したい場合は、
        # 辞書の値を別の名前に更新することで変更可能です。
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END}, # path_map
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part2.png")

    user_input = "「推しの子」ってどんな話ですか？"
    config={"run_name": "(part 2) Simple agent with a tool"}
    result = graph.invoke({"messages": [("user", user_input)]}, 
                            config=config)
    
    ic(type(result['messages'][-1].content)) # str
    print(result['messages'][-1].content)

# persistent checkpointing によってメモリ機能を実装する
def main_part3():
    from langgraph.checkpoint.memory import MemorySaver

    # MemorySaver は開発用途でのみ使うのがよい。
    # リリースする段階では langgraph-checkpoint-postgres と PostgresSaver を使うことを推奨。
    # Ref https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver
    memory = MemorySaver()

    from typing import Annotated

    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import BaseMessage
    from typing_extensions import TypedDict

    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    tools = [tool]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)


    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}


    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # コンパイル時に checkpointer を渡す
    graph = graph_builder.compile(checkpointer=memory)

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part3.png")

    # 会話のキーとして使用するスレッドIDを与える
    config = {"configurable": {"thread_id": "1"}, "run_name": "(part 3) Add memory"}

    user_input = "こんにちは！私の名前はkossyです！"

    # graph を呼ぶときに thread_id を渡す
    # config は2番目の位置引数であり、1番目の辞書の中ではないことに注意
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    print(result['messages'][-1].content)

    # 記憶していることを確かめるクエリ
    user_input = "私の名前を覚えていますか？"
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    print(result['messages'][-1].content)

    # わざと thread_id を変えてみる
    config = {"configurable": {"thread_id": "2"}, "run_name": "(part 3) Add memory (thread_id = 2)"}
    user_input = "私の名前を覚えていますか？"
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    print(result['messages'][-1].content)
    
    # チェックポイントに何が含まれるかを確認する
    config = {"configurable": {"thread_id": "1"}}
    snapshot_id1 = graph.get_state(config)
    ic(snapshot_id1.values['messages'])
    ic(snapshot_id1.next) # ENDステートに到達しているので next は空

    # チェックポイントに何が含まれるかを確認する
    config = {"configurable": {"thread_id": "2"}}
    snapshot_id2 = graph.get_state(config)
    ic(snapshot_id2.values['messages'])
    ic(snapshot_id2.next) # ENDステートに到達しているので next は空

# human-in-the-loop の実装
def main_part4():
    from langgraph.checkpoint.memory import MemorySaver
    from typing import Annotated
    from langchain_core.messages import BaseMessage
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    memory = MemorySaver()

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    tools = [tool]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # ================== ここまでは part3 と同じ ==================

    graph = graph_builder.compile(
        checkpointer=memory,
        interrupt_before=["tools"], # 追加
        # Note: can also interrupt __after__ tools, if desired.
        # interrupt_after=["tools"]
    )

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part4.png")

    user_input = "Langchain の勉強をしています。 Langchain に関する情報を検索してもらえませんか？"
    config = {"configurable": {"thread_id": "1"}, "run_name": "(part 4) Human-in-the-loop"}
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    ic(result['messages'][-1].tool_calls)
    # [{'args': {'query': 'Langchain'},
    #   'id': 'call_e0r6Iog28zleirLUyRKCD8VH',
    #   'name': 'wikipedia',
    #   'type': 'tool_call'}]
    
    snapshot = graph.get_state(config)
    ic(snapshot.values['messages'])
    ic(len(snapshot.values['messages'])) # 2 (HumanMessage と AIMessage)
    ic(snapshot.next) # ENDステートに到達しているので next は空

    existing_message = snapshot.values["messages"][-1]
    ic(existing_message.tool_calls)
    # ic| existing_message.tool_calls: [{'args': {'query': 'Langchain'},
    #                                    'id': 'call_SIdeSwqnol7mgAybd8RH7hEU',
    #                                    'name': 'wikipedia',
    #                                    'type': 'tool_call'}]

    # None を指定すると、中断したところからグラフを続行できる
    result = graph.invoke(None, config)
    ic(result['messages'][-1].content)

# 手動でルーティングする方法
def main_part5():
    from langgraph.checkpoint.memory import MemorySaver
    from typing import Annotated
    from langchain_core.messages import BaseMessage
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition

    memory = MemorySaver()

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    tools = [tool]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        # This is new!
        interrupt_before=["tools"],
        # Note: can also interrupt **after** actions, if desired.
        # interrupt_after=["tools"]
    )

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part5.png")

    user_input = "Langgraph の勉強をしています。Langgraph に関する情報を検索してもらえませんか？"
    config = {"configurable": {"thread_id": "1"}, "run_name": "(part 5) Manually Updating the State"}
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    ic(result['messages'][-1].tool_calls)

    # ================== ここまでは part4 と同じ ==================

    # ツールが呼ばれたところで中断
    snapshot = graph.get_state(config)
    existing_message = snapshot.values["messages"][-1]
    existing_message.pretty_print()

    from langchain_core.messages import AIMessage, ToolMessage

    answer = (
        "LangGraphは、LLM(大規模言語モデル)を使用して状態を持つマルチアクターアプリケーションを構築するためのライブラリです。"
    )

    # ツールとAIの応答をこちらで生成して渡す
    new_messages = [
        # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
        ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
        # And then directly "put words in the LLM's mouth" by populating its response.
        AIMessage(content=answer),
    ]

    # update_state 関数で state を更新する
    graph.update_state(
        # Which state to update
        config,
        # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
        # to the existing state. We will review how to update existing messages in the next section!
        {"messages": new_messages},
    )

    snapshot = graph.get_state(config)
    ic(snapshot.values["messages"])

    # "chatbot" ノードから来たかのように扱う
    graph.update_state(
        config,
        {"messages": [AIMessage(content="私はAIのエキスパートです！")]},
        # Which node for this function to act as. It will automatically continue
        # processing as if this node just ran.
        as_node="chatbot",
    )

    snapshot = graph.get_state(config)
    ic(snapshot.values['messages'][-3:]) # [ToolMessage, AIMessage, AIMessage]
    ic(snapshot.next) # 空

    # メッセージを上書きしたい場合は？
    # -> 既存のメッセージのIDと同じメッセージを作成し、それを使ってupdate_stateを行う。
    user_input = "Langgraph の勉強をしています。Langgraph に関する情報を検索してもらえませんか？"
    config = {"configurable": {"thread_id": "2"}, "run_name": "(part 5) Manually Updating the State"}
    result = graph.invoke({"messages": [("user", user_input)]}, config)

    from langchain_core.messages import AIMessage

    snapshot = graph.get_state(config)
    existing_message = snapshot.values["messages"][-1]
    ic(existing_message.tool_calls[0]) # 元々は "LangGraph" が呼ばれる予定だった
    """
    ic| existing_message.tool_calls[0]: {'args': {'query': 'Langgraph'},
                                        'id': 'call_ipHOPq9fbeM1SqPrkJWmqpt7',
                                        'name': 'wikipedia',
                                        'type': 'tool_call'}
    """

    new_tool_call = existing_message.tool_calls[0].copy()
    new_tool_call["args"]["query"] = "LangChain" # 変更した

    new_message = AIMessage(
        content=existing_message.content,
        tool_calls=[new_tool_call],
        id=existing_message.id, # 重要: IDが一致していないと置換できない
    )

    graph.update_state(config, {"messages": [new_message]})

    snapshot = graph.get_state(config)
    ic(snapshot.values['messages']) # [Human, AI]
    ic(snapshot.next) # ('tools',)

    result = graph.invoke(None, config)
    print(f"final output: {result['messages'][-1].content}")
    """
    final output: "LangChain"は、大規模な言語モデル（LLM）をアプリケーションに統合するためのソフトウェアフレームワークです。このフレーム ワークは、文書の分析や要約、チャットボット、コードの分析など、言語モデルの一般的な使用ケースと大きく重なります。

    Langgraphについての特定の情報は見つかりませんでしたが、LangChainがその一部か関連する技術である可能性が考えられます。もしさらに詳細な 情報や特定の側面について知りたい場合、追加の情報を教えてください。
    """

    snapshot = graph.get_state(config)
    ic(snapshot.values['messages']) # [Human, AI, Tool, AI]
    ic(snapshot.next) # ()

# State をカスタマイズする
def main_part6():
    from langgraph.checkpoint.memory import MemorySaver
    from typing import Annotated
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        ask_human: bool # 追加

    # Pydantic v2 の BaseModel を使用する。
    # これには langchain-core >= 0.3 が必要。
    # langchain-core < 0.3 だと Pydantic v1 と v2 の BaseModel が混在することでエラーが発生する
    from pydantic import BaseModel

    class RequestAssistance(BaseModel):
        """会話を専門家にエスカレーションする。これは、あなたが直接支援できない場合や、ユーザーがあなたの権限を超えたサポートを必要とする場合に使用します。

        この機能を使用するには、ユーザーの 'request' を専門家に伝えて、適切なガイダンスを提供できるようにしてください。
        """

        request: str

    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    tools = [tool]
    llm = ChatOpenAI(model="gpt-4o")
    # We can bind the llm to a tool definition, a pydantic model, or a json schema
    llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


    def chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        ask_human = False
        # RequestAssistance フラグが呼ばれたら ask_human フラグを立てる
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True
        return {"messages": [response], "ask_human": ask_human}

    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=[tool]))

    from langchain_core.messages import AIMessage, ToolMessage


    def create_response(response: str, ai_message: AIMessage):
        return ToolMessage(
            content=response,
            tool_call_id=ai_message.tool_calls[0]["id"],
        )


    def human_node(state: State):
        new_messages = []
        if not isinstance(state["messages"][-1], ToolMessage):
            # Typically, the user will have updated the state during the interrupt.
            # If they choose not to, we will include a placeholder ToolMessage to
            # let the LLM continue.
            new_messages.append(
                create_response("No response from human.", state["messages"][-1])
            )
        return {
            # Append the new messages
            "messages": new_messages,
            # Unset the flag
            "ask_human": False,
        }


    graph_builder.add_node("human", human_node)

    def select_next_node(state: State):
        if state["ask_human"]:
            return "human"
        # Otherwise, we can route as before
        return tools_condition(state)


    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", END: END},
    )

    # The rest is the same
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        # We interrupt before 'human' here instead.
        interrupt_before=["human"],
    )

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part6.png")

    user_input = "AIエージェントを構築するために専門家のサポートが必要です。あなたに専門家を要求します。"
    config = {"configurable": {"thread_id": "1"}, "run_name": "(part 6) Customizing state"}    
    result = graph.invoke({"messages": [("user", user_input)]}, config)
    ic(result['messages'][-1].tool_calls)
    """
    [{'args': {'request': 'AIエージェントを構築するためのサポートが必要です。'},
    'id': 'call_DZrEIWS36e3MGh9OW9ROjcBF',
    'name': 'RequestAssistance',
    'type': 'tool_call'}]
    """
    
    snapshot = graph.get_state(config)
    ic(snapshot.next) # ('human',)

    ai_message = snapshot.values["messages"][-1]
    human_response = (
        "ご指名ありがとうございます。我々AIエージェント構築専門家が対応いたします。"
        " LangGraphを使うことで、シンプルな自律型エージェントよりも、はるかに信頼性が高く拡張性のあるエージェントを構築できますよ。"
    )
    tool_message = create_response(human_response, ai_message)
    graph.update_state(config, {"messages": [tool_message]})

    ic(graph.get_state(config).values["messages"])
    # [HumanMessage(content='AIエージェントを構築するために専門家のサポートが必要です。あなたに専門家を要求します。', additional_kwargs={}, response_metadata={}, id='dfa4519c-8deb-48ea-bb45-b0cc7386ef8f'),
    # AIMessage(content='', ..., tool_calls=[{'name': 'RequestAssistance', 'args': {'request': 'AIエージェントを構築するために専門家のサポートが必要です。'}, 'id': 'call_nt2BCIvoqhouwWNAAKxkaclC', 'type': 'tool_call'}], ...),
    # ToolMessage(content='ご指名ありがとうございます。我々AIエージェント構築専門家が対応 いたします。 LangGraphを使うことで、シンプルな自律型エージェントよりも、はるかに信頼性が高く拡張性のあるエージェントを構築できますよ 。', id='105d4bc9-6c8c-4a61-bdcd-7845b63f2da8', tool_call_id='call_nt2BCIvoqhouwWNAAKxkaclC')]

    # 中断したところから再開(human ノードへ)
    result = graph.invoke(None, config)

# 過去のやり取りに戻る方法
def main_part7():
    from langgraph.checkpoint.memory import MemorySaver
    from typing import Annotated
    from langchain_core.messages import BaseMessage
    from typing_extensions import TypedDict
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode, tools_condition
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        ask_human: bool # 追加

    # Pydantic v2 の BaseModel を使用する。
    # これには langchain-core >= 0.3 が必要。
    # langchain-core < 0.3 だと Pydantic v1 と v2 の BaseModel が混在することでエラーが発生する
    from pydantic import BaseModel

    class RequestAssistance(BaseModel):
        """会話を専門家にエスカレーションする。これは、あなたが直接支援できない場合や、ユーザーがあなたの権限を超えたサポートを必要とする場合に使用します。

        この機能を使用するには、ユーザーの 'request' を専門家に伝えて、適切なガイダンスを提供できるようにしてください。
        """

        request: str

    tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))
    tools = [tool]
    llm = ChatOpenAI(model="gpt-4o")
    # We can bind the llm to a tool definition, a pydantic model, or a json schema
    llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


    def chatbot(state: State):
        response = llm_with_tools.invoke(state["messages"])
        ask_human = False
        # RequestAssistance フラグが呼ばれたら ask_human フラグを立てる
        if (
            response.tool_calls
            and response.tool_calls[0]["name"] == RequestAssistance.__name__
        ):
            ask_human = True
        return {"messages": [response], "ask_human": ask_human}

    graph_builder = StateGraph(State)

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools=[tool]))

    from langchain_core.messages import AIMessage, ToolMessage


    def create_response(response: str, ai_message: AIMessage):
        return ToolMessage(
            content=response,
            tool_call_id=ai_message.tool_calls[0]["id"],
        )


    def human_node(state: State):
        new_messages = []
        if not isinstance(state["messages"][-1], ToolMessage):
            # Typically, the user will have updated the state during the interrupt.
            # If they choose not to, we will include a placeholder ToolMessage to
            # let the LLM continue.
            new_messages.append(
                create_response("No response from human.", state["messages"][-1])
            )
        return {
            # Append the new messages
            "messages": new_messages,
            # Unset the flag
            "ask_human": False,
        }


    graph_builder.add_node("human", human_node)

    def select_next_node(state: State):
        if state["ask_human"]:
            return "human"
        # Otherwise, we can route as before
        return tools_condition(state)


    graph_builder.add_conditional_edges(
        "chatbot",
        select_next_node,
        {"human": "human", "tools": "tools", END: END},
    )

    # The rest is the same
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge("human", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    memory = MemorySaver()
    graph = graph_builder.compile(
        checkpointer=memory,
        # We interrupt before 'human' here instead.
        interrupt_before=["human"],
    )

    # ASCII表示で確認
    print(graph.get_graph().draw_ascii()) 

    # グラフを画像にして確認
    from utils import save_compiled_state_graph
    save_compiled_state_graph(graph, png_filename="img/quickstart-part7.png")

    # ============================== ここまで part6 と同じ ==============================

    config = {"configurable": {"thread_id": "1"}, "run_name": "(part 7) Time travel"}
    response = graph.invoke(
        {
            "messages": [
                ("user", "Langchainって何ですか？検索してほしいです。")
            ]
        },
        config,
    )

    response = graph.invoke(
        {
            "messages": [
                ("user", "便利なフレームワークですね！そもそも言語モデルって何ですか？")
            ]
        },
        config,
    )
    
    # リプレイしてみます
    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 6:
            to_replay = state
    """
    Num Messages:  8 Next:  ()
    --------------------------------------------------------------------------------
    Num Messages:  7 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  6 Next:  ('tools',)
    --------------------------------------------------------------------------------
    Num Messages:  5 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  4 Next:  ('__start__',)
    --------------------------------------------------------------------------------
    Num Messages:  4 Next:  ()
    --------------------------------------------------------------------------------
    Num Messages:  3 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  2 Next:  ('tools',)
    --------------------------------------------------------------------------------
    Num Messages:  1 Next:  ('chatbot',)
    --------------------------------------------------------------------------------
    Num Messages:  0 Next:  ('__start__',)
    --------------------------------------------------------------------------------
    """

    ic(to_replay.next) # ('tools',)
    ic(to_replay.config)
    #  {'configurable': {'checkpoint_id': '1efad733-46fd-6963-8006-11c4720fe5cc', # タイムスタンプがある
    #                    'checkpoint_ns': '',
    #                    'thread_id': '1'}}

    # The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
    response = graph.invoke(None, to_replay.config)
    ic(response['messages'][-1].content)

if __name__ == "__main__":
    # シンプルな chatbot の構築
    main_part1()

    # ツールを追加してシンプルな Agent を実装する
    main_part2()

    # persistent checkpointing によってメモリ機能を実装する
    main_part3()

    # human-in-the-loop の実装
    main_part4()

    # 手動で state を更新する
    main_part5()

    # State をカスタマイズする
    main_part6()

    # 過去のやり取りに戻る方法
    main_part7()
