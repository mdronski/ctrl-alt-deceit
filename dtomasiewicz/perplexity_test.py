import getpass
import os

if not os.environ.get("PPLX_API_KEY"):
  os.environ["PPLX_API_KEY"] = getpass.getpass("Enter API key for Perplexity: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("sonar", model_provider="perplexity")

prompt = input("Enter a prompt\n")

response = model.invoke(prompt)
print(response.content)
print("Finished")