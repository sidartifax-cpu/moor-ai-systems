from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("Loading PDF...")
loader = PyPDFLoader("../MAAT_The_5_Traps.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks")

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("Vector store created")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "You are a MAAT advisor. Answer using only the context below.\n"
    "If the answer is not in the context, say: That knowledge is not in this scroll.\n\n"
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
print("=" * 50)

questions = [
    "What is the Scarcity Tax and what is the MAAT solution?",
    "How does Social Gravity trap people?",
    "What is the Never-Sell Doctrine?"
]

for q in questions:
    print(f"Question: {q}")
    answer = chain.invoke(q)
    print(f"Answer: {answer}")
    print("-" * 50)
