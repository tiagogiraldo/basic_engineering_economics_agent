import gradio as gr
from langchain_core.messages import HumanMessage, messages_to_dict
from langgraph.graph import StateGraph, END
from agents.agents_nodes import agent_node, format_output, tool_node
from utils.state_utils import AgentState

def create_interface():

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.add_node("format", format_output)
    
    graph.set_entry_point("agent")
    graph.add_edge("agent", "tool")
    graph.add_edge("tool", "format")
    graph.add_edge("format", END)
    
    app = graph.compile()

    def process_query(query: str) -> dict:
        try:
            inputs = {"messages": [HumanMessage(content=query)]}
            result = app.invoke(inputs)  
            return messages_to_dict(result['messages'])[2]['data']['content']
        except Exception as e:
            return {"error": f"Execution error: {str(e)}"}


    with gr.Blocks(title="Time Value of Money Calculator") as interface:
        gr.Markdown("##  Time Value of Money Calculator")
        gr.Markdown("Enter natural language queries about present/future value calculations")
        
        with gr.Row():
            input_text = gr.Textbox(
                label="Financial Question",
                placeholder="E.g.: Present value of $3000 in 5 years at 8% interest?",
                lines=3
            )
            output_json = gr.JSON(label="Result")  
        
        submit_btn = gr.Button("Calculate")
        submit_btn.click(
            fn=process_query,       
            inputs=input_text,      
            outputs=output_json     
        )
    return interface
