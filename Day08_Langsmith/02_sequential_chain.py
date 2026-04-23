from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
prompt1=PromptTemplate(
    template='Genrate the detailed report for the following topic:{topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Genrate the detailed 5 point summary for the following text:{text}',
    input_variables=['text']
)
model=ChatOpenAI(model="gpt-4o-mini")
parser=StrOutputParser()
chain=prompt1 | model| parser | prompt2 | model | parser
result =chain.invoke({"topic":"What is the capital of France?"})
print(result)