import getpass
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

open_ai_key = os.getenv("OPEN_AI_KEY")

# Check if the key is loaded
if open_ai_key is None:
    raise ValueError("OPENAI_AI_KEY environment variable is not set.")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4", api_key=open_ai_key)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "ice cream"})

prompt_value = prompt.invoke({"topic": "ice cream"})

print(prompt_value.to_string())

message = model.invoke(prompt_value)

print(output_parser.invoke(message))