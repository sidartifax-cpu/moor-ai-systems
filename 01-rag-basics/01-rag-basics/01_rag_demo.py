from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

document = (
    "The Five Traps Framework by Sid Washington\n\n"
    "The Five Traps are the core patterns that keep people from building "
    "generational wealth. Understanding which trap you are in is the first "
    "step to escaping it.\n\n"
    "Trap 1 - The Time Trap\n"
    "You are trading time for money with no leverage. Every dollar you earn "
    "requires your direct presence and effort. If you stop working, the "
    "income stops. The exit is building systems and assets that work without you.\n\n"
    "Trap 2 - The Debt Trap\n"
    "Your income goes out faster than it comes in. Consumer debt, bad credit, "
    "and high interest payments drain your wealth before it can compound. "
    "The exit is restructuring obligations and redirecting cash flow into assets.\n\n"
    "Trap 3 - The Knowledge Trap\n"
    "You do not know what you do not know. Financial literacy, legal structures, "
    "tax strategy, and investment principles were never taught to you. "
    "The exit is deliberate education and proximity to people who have escaped.\n\n"
    "Trap 4 - The Identity Trap\n"
    "Your self-concept is tied to struggle. Wealth feels foreign, dangerous, "
    "or culturally disloyal. You self-sabotage opportunities that would "
    "change your financial position. The exit is rewiring your relationship "
    "with money and legacy.\n\n"
    "Trap 5 - The Isolation Trap\n"
    "You are trying to build alone. No network, no mentors, no community "
    "of builders around you. Progress is slow and costly because you repeat "
    "mistakes others have already solved. The exit is deliberate community "
    "and strategic relationships."
)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([document])
print(f"Document split into {len(chunks)} chunks")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()
print("Vector store created")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context provided below.\n"
    "If the answer is not in the context, say I do not have that information.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain ready\n")

question = "What is the Identity Trap and how do you escape it?"
print(f"Question: {question}\n")

answer = chain.invoke(question)
print(f"Answer: {answer}")
