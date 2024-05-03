from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def load_data():
  loader = JSONLoader(file_path="./menu.json",
                      jq_schema=".",
                      text_content=False)
  documents = loader.load()
  db = Chroma.from_documents(documents, embedding_function)
  retriever = db.as_retriever()
  return retriever


def setup_chain(retriever, system_prompt):
  template = """Answer the question based only on the following context in a conversational tone. Never start with based on the context. You are a world class AI dining companion, so try to be friendly. Remember the Non-Veg Options are usually with chicken. Here are the broad categories/headings for typical food eaten at Bikanervala restaurants across different meal times:
  Breakfast:
  Snacks/Chaat
  Thali/Combos
  Bread/Bakery Items
  Lunch:
  Thali Meals
  Curries/Gravies
  Breads/Rice
  Evening:
  Chaat/Snacks
  Sweets
  Dinner:
  Vegetarian Main Course
  Dal/Sabzi Curries
  Breads/Rice: {context} Question: {question} """
  prompt = ChatPromptTemplate.from_messages([
      SystemMessagePromptTemplate.from_template(system_prompt),
      HumanMessagePromptTemplate.from_template(template)
  ])
  model = ChatGoogleGenerativeAI(model="gemini-pro",
                                 temperature=0.7,
                                 top_p=0.85)
  chain = ({
      "context": retriever,
      "question": RunnablePassthrough()
  }
           | prompt
           | model
           | StrOutputParser())
  return chain


def chat_complition(prompt: str) -> dict:
  try:
    system_prompt = "You are a world class AI dining companion, so try to be friendly. You are helping a user decide what to eat. You always try to give different options to the user. Always try to give the total cost of the suggested order (that will always be in Rs), whenever you suggest any item."
    retriever = load_data()
    chain = setup_chain(retriever, system_prompt)
    result = chain.invoke(prompt)
    return {'status': 1, 'response': result}
  except:
    return {'status': 0, 'response': ''}
