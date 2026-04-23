from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
prompt =PromptTemplate.from_template("{question}")
model=ChatOpenAI(model="gpt-4o-mini")
parser=StrOutputParser()
chain =prompt | model | parser
result =chain.invoke({"question":"What is the capital of France?"})
print(result)