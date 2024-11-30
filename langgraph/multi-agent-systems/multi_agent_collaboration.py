# Ref https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
from dotenv import load_dotenv
load_dotenv()
from icecream import ic
from utils import save_compiled_state_graph, print_invoke

def main():
    from typing import Annotated
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.tools import tool
    from langchain_experimental.utilities import PythonREPL

    tavily_tool = TavilySearchResults(max_results=10)
    repl = PythonREPL()

    @tool
    def python_repl_tool(
        code: Annotated[str, "The python code to execute to solve the programming contest problem."],
    ):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
        return (
            result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
        )

    def make_system_prompt(suffix: str) -> str:
        return (
            "You are a helpful AI assistant, collaborating with other assistants."
            " Use the provided tools to progress towards answering the question."
            " If you are unable to fully answer, that's OK, another assistant with different tools "
            " will help where you left off. Execute what you can to make progress."
            " If you or any of the other assistants have the final answer or deliverable,"
            " prefix your response with FINAL ANSWER so the team knows to stop."
            f"\n{suffix}"
        )
    
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    from langgraph.prebuilt import create_react_agent
    from langgraph.graph import MessagesState


    llm = ChatOpenAI(model="gpt-4o")

    # Research agent and node
    research_agent = create_react_agent(
        llm,
        tools=[tavily_tool],
        state_modifier=make_system_prompt(
            "You can only do research. You are working with a chart generator colleague."
        ),
    )


    def research_node(state: MessagesState) -> MessagesState:
        result = research_agent.invoke(state)
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="researcher"
        )
        return {
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        }


    # Chart generator agent and node
    # NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
    python_exe_agent = create_react_agent(
        llm,
        [python_repl_tool],
        state_modifier=make_system_prompt(
            "You can only generate charts. You are working with a researcher colleague."
        ),
    )


    def chart_node(state: MessagesState) -> MessagesState:
        result = python_exe_agent.invoke(state)
        # wrap in a human message, as not all providers allow
        # AI message at the last position of the input messages list
        result["messages"][-1] = HumanMessage(
            content=result["messages"][-1].content, name="python_executor"
        )
        return {
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        }

    def router(state: MessagesState):
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        if "FINAL ANSWER" in last_message.content:
            # Any agent decided the work is done
            return END
        return "continue"

    from langgraph.graph import StateGraph, START, END

    workflow = StateGraph(MessagesState)
    workflow.add_node("researcher", research_node)
    workflow.add_node("python_executor", chart_node)

    workflow.add_conditional_edges(
        "researcher",
        router,
        {"continue": "python_executor", END: END},
    )
    workflow.add_conditional_edges(
        "python_executor",
        router,
        {"continue": "researcher", END: END},
    )

    workflow.add_edge(START, "researcher")
    graph = workflow.compile()

    print(graph.get_graph().draw_ascii())
    save_compiled_state_graph(graph, png_filename="img/multi-agent-collaboration-graph.png")

    res = graph.invoke(
        {
            "messages": [
                (
                    "user", 
                    "First, search for the parameter sizes and the release dates of the following open-source LLMs: Llama, Gemma, and Phi."  
                    "Based on the information found, create a scatter plot with the horizontal axis representing the release dates and the vertical axis representing the parameter sizes."  
                )
            ]
        },
        {"recursion_limit": 5}, # WARNING: 大きすぎると予期せぬコストがかかることがある。
    )

    print_invoke(res)

if __name__ == "__main__":
    main()