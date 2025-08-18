import gradio as gr
import time
import dotenv

from app.agent import agent


def ask_agent(company_name):
    # TODO conduct valid prompt engineering
    user_input = f"What are potential red flag about company called {company_name} as a potential business partner. Prepare a markdown summary. Provide links and informations you got from tools"
    response = agent.invoke({"messages": user_input})
    return response["messages"][-1].content


with gr.Blocks() as app:
    gr.Markdown("## Company Application Processor")

    company_name = gr.Textbox(label="Company name")
    start_button = gr.Button("Start information gathering")

    summary = gr.Markdown("**Summary**")
    result = gr.Markdown("", container=True)

    start_button.click(fn=ask_agent, inputs=[company_name], outputs=[result])

def main():
    app.launch()

if __name__ == "__main__":
    main()