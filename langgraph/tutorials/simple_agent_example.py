# Ref https://langchain-ai.github.io/langgraph/#example
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage

# ChatAnthropicの代わりにChatOpenAIを使う
from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # 実際に検索していないことはLLMにナイショにしておいてね
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


tools = [search]

tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

# Define the function that determines whether to continue or not
# Literal[...]は戻り値がカッコ内のどちらかでなければならないことを示す
# ただし、インタプリタが型チェックをする訳ではないので、違反しても実行時エラーはでない
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node(node="agent", action=call_model) # 次の行動を決める責任を負う
workflow.add_node(node="tools", action=tool_node)  # toolを呼ぶ

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
# conditional edgeとは、グラフのMessageStateの内容に依存して次の行き先が決まる辺のこと
workflow.add_conditional_edges(
    # この辺(edge)はagentノードが呼ばれた後に使われる
    "agent",
    # 次に呼び出されるノードを決める関数をここで渡す
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# # app はCompiledStateGraph型で、Runnableを継承している。
# print(f"[DEBUG] type(app): {type(app)}") # <class 'langgraph.graph.state.CompiledStateGraph'>

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
print(final_state["messages"][-1].content)
# output: The current weather in San Francisco is 60 degrees and foggy.

# print(f"[DEBUG] type(final_state): {type(final_state)}") # <class 'langgraph.pregel.io.AddableValuesDict'>
# print(f"[DEBUG] final_state: {final_state}")

# final_stateの中身
"""
{'messages': 
    [
        HumanMessage(content='what is the weather in sf', additional_kwargs={}, response_metadata={}, id=(omitted)), 
        AIMessage(
            content='', 
            additional_kwargs=
                {'tool_calls': 
                    [
                        {'id': (omitted), 
                        'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'search'}, 
                        'type': 'function'}
                    ], 
                    'refusal': None
                }, 
            response_metadata=
                {'token_usage': 
                    {'completion_tokens': 17, 'prompt_tokens': 48, 'total_tokens': 65, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}
                    }, 
                'model_name': 'gpt-4o-2024-08-06', 
                'system_fingerprint': (omitted), 
                'finish_reason': 'tool_calls', 
                'logprobs': None
                }, 
            id=(omitted), 
            tool_calls=
                [
                    {'name': 'search', 
                    'args': {'query': 'current weather in San Francisco'}, 
                    'id': (omitted), 
                    'type': 'tool_call'}
                ], 
            usage_metadata=
                {'input_tokens': 48, 
                'output_tokens': 17, 
                'total_tokens': 65, 
                'input_token_details': {'cache_read': 0}, 
                'output_token_details': {'reasoning': 0}
                }
        ), 
        ToolMessage(
            content="It's 60 degrees and foggy.", 
            name='search', 
            id=(omitted), 
            tool_call_id=(omitted)
        ), 
        AIMessage(
            content='The current weather in San Francisco is 60 degrees and foggy.', 
            additional_kwargs={'refusal': None}, 
            response_metadata={'token_usage': {'completion_tokens': 15, 'prompt_tokens': 80, 'total_tokens': 95, 'completion_tokens_details': {'audio_tokens': None, 'reasoning_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': (omitted), 'finish_reason': 'stop', 'logprobs': None}, id=(omitted), usage_metadata={'input_tokens': 80, 'output_tokens': 15, 'total_tokens': 95, 'input_token_details': {'cache_read': 0}, 'output_token_details': {'reasoning': 0}}
        )
    ]
}
"""

# おまけ: graphをmermaidで記述した画像を保存する
from utils import save_compiled_state_graph
save_compiled_state_graph(app)