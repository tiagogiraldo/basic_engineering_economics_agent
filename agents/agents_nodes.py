import json
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from utils.state_utils import AgentState
from tools.financial_tools import time_value_tool
from langchain_ollama import ChatOllama


# LLL instantation

llm = ChatOllama(model="qwen3:4b", temperature=0)
llm_instantiated = llm.bind_tools(    
    [time_value_tool],
    tool_choice={"type": "function", "function": {"name": "time_value_tool"}}
)

def agent_node(state: AgentState):
    response = llm_instantiated.invoke(state["messages"])
    if not (hasattr(response, 'tool_calls') and response.tool_calls):
        error_message = AIMessage(content="Error: Model failed to generate tool call.")
        return {"messages": [error_message]}
    return {"messages": [response]}

# Tool node executes the tool
tool_node = ToolNode([time_value_tool])

# Factor to output mapping
F_MAPPING = {
    "P/F": "PV", "P/A": "PV", "P/G": "PV",
    "F/P": "FV", "F/A": "FV", "F/G": "FV",
    "A/P": "Annual", "A/F": "Annual", "A/G": "Annual"
}

def format_output(state: AgentState):
    try:
        # The last message should be the ToolMessage (from the tool node)
        if not state["messages"] or not isinstance(state["messages"][-1], ToolMessage):
            return {"output": {"error": "No tool result found in the last message"}}
        
        tool_message = state["messages"][-1]
        # Parse the content of the tool message as JSON
        tool_result = json.loads(tool_message.content)
        
        # The second last message should be the AIMessage with the tool call
        if len(state["messages"]) < 2 or not isinstance(state["messages"][-2], AIMessage):
            return {"output": {"error": "No AI message (with tool call) found before the tool message"}}
        
        ai_message = state["messages"][-2]
        if not ai_message.tool_calls:
            return {"output": {"error": "The AI message does not contain tool calls"}}
        
        # We take the first tool call (since we forced one tool)
        tool_call = ai_message.tool_calls
        args = tool_call["args"]
        
        # Get the factor type from the args
        factor_type = args["F"]
        if factor_type not in F_MAPPING:
            return {"output": {"error": f"Unrecognized factor type: {factor_type}"}}
        
        result_key = F_MAPPING[factor_type]
        if result_key not in tool_result:
            return {"output": {"error": f"Expected key {result_key} not found in tool result"}}
        
        value = tool_result[result_key]
        return {"output": {result_key: round(float(value), 2)}}
        
    except (KeyError, TypeError, json.JSONDecodeError, IndexError) as e:
        return {"output": {"error": f"Result formatting failed: {str(e)}"}}