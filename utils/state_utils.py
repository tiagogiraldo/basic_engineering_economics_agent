from typing import TypedDict, Annotated, Union, List, Dict
from langchain_core.messages import BaseMessage
import operator


# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    output: Union[Dict, None] = None