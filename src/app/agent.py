from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model

from app.tools import TOOLS

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
agent = create_react_agent(
    tools=TOOLS,
    model=llm,
)