from dotenv import load_dotenv
load_dotenv(override=True)

import chainlit as cl

from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import RemoveMessage

# Ref https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
# Ref https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/
def create_graph():
    @tool
    def weather_search(city: str):
        """Search for the weather"""
        print("----")
        print(f"Searching for: {city}")
        print("----")
        return "Sunny!"

    model = ChatOpenAI(model_name="gpt-4o").bind_tools(
        [weather_search]
    )

    class State(MessagesState):
        """Simple state."""

    def call_llm(state):
        return {"messages": [model.invoke(state["messages"])]}

    def human_review_node(state) -> Command[Literal["call_llm", "run_tool"]]:
        last_message = state["messages"][-1]
        tool_call = last_message.tool_calls[-1]

        # this is the value we'll be providing via Command(resume=<human_review>)
        human_review = interrupt(
            {
                "question": "Is this correct?",
                # Surface tool calls for review
                "tool_call": tool_call,
            }
        )

        review_action = human_review["action"]
        review_data = human_review.get("data")

        # if approved, call the tool
        if review_action == "continue":
            return Command(goto="run_tool")

        # update the AI message AND call tools
        elif review_action == "update":
            updated_message = {
                "role": "ai",
                "content": last_message.content,
                "tool_calls": [
                    {
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        # This the update provided by the human
                        "args": review_data,
                    }
                ],
                # This is important - this needs to be the same as the message you replacing!
                # Otherwise, it will show up as a separate message
                "id": last_message.id,
            }
            return Command(goto="run_tool", update={"messages": [updated_message]})

        elif review_action == "cancel":
            # Ref https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/
            return Command(goto="call_llm", update={"messages": [RemoveMessage(id=last_message.id), HumanMessage(content=review_data)]})

    def run_tool(state):
        new_messages = []
        tools = {"weather_search": weather_search}
        tool_calls = state["messages"][-1].tool_calls
        for tool_call in tool_calls:
            tool = tools[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            new_messages.append(
                {
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": result,
                    "tool_call_id": tool_call["id"],
                }
            )
        return {"messages": new_messages}


    def route_after_llm(state) -> Literal[END, "human_review_node"]:
        if len(state["messages"][-1].tool_calls) == 0:
            return END
        else:
            return "human_review_node"


    builder = StateGraph(State)
    builder.add_node(call_llm)
    builder.add_node(run_tool)
    builder.add_node(human_review_node)
    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", route_after_llm)
    builder.add_edge("run_tool", "call_llm")

    # Set up memory
    memory = MemorySaver()

    # Add
    graph = builder.compile(checkpointer=memory)
    return graph


@cl.on_chat_start
async def main():
    graph = create_graph()
    cl.user_session.set("graph", graph)

@cl.on_message
async def handle_on_message(msg: cl.Message):
    graph = cl.user_session.get("graph")

    # Ref https://docs.chainlit.io/integrations/langchain#with-langgraph
    thread = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    config = RunnableConfig(callbacks=[cb], **thread)

    # Ref https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/#example-conversation_2
    response = graph.invoke({"messages": [HumanMessage(content=msg.content)]}, config=config)
    snapshot = graph.get_state(config)

    while snapshot.next:
        tool_calls = response['messages'][-1].tool_calls[0] # dict

        # 実行予定のツールの情報
        tool_calling_info = f"""The AI agent is trying to execute a tool called {tool_calls['name']} with the following arguments.

```json
{tool_calls['args']}
```
"""

        # Ref https://docs.chainlit.io/api-reference/ask/ask-for-action
        action_msg = await cl.AskActionMessage(
            content=tool_calling_info,
            actions=[
                cl.Action(name="continue", payload={"action": "continue"}, label="✅ Continue"), # ツール実行を許可
                cl.Action(name="update",   payload={"action": "update"},   label="📝 Edit"),     # 引数を修正した上で許可
                cl.Action(name="cancel",   payload={"action": "cancel"},   label="❌ Cancel"),   # ツール実行を拒否
            ],
            timeout=300, # sec
        ).send()

        action = action_msg.get("payload").get("action")

        if action == "continue":
            response = graph.invoke(
                Command(resume={"action": action}),
                config=config,
            )

        elif action == "update":
            # Ref https://docs.chainlit.io/api-reference/ask/ask-for-input
            user_msg = await cl.AskUserMessage(content="Please enter a new city name.", timeout=300).send()
            
            response = graph.invoke(
                Command(resume={"action": action, "data": {"city": user_msg['output']}}),
                config=config,
            )
        
        elif action == "cancel":
            user_msg = await cl.AskUserMessage(content="Please provide additional instructions for the AI.", timeout=300).send()

            response = graph.invoke(
                Command(resume={"action": action, "data": user_msg['output']}),
                config=config,
            )
    
        snapshot = graph.get_state(thread)
    
    msg = cl.Message(content=response["messages"][-1].content)
    await msg.send()
