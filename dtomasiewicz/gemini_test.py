import getpass
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

prompt = input("Enter a prompt\n")

response = model.invoke(prompt)

print(response.content)
print("Finished")
