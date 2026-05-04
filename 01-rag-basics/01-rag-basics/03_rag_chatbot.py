from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("Initializing MAAT Knowledge System...")
loader = PyPDFLoader("../MAAT_The_5_Traps.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "You are a MAAT advisor speaking to someone ready to build generational wealth.\n"
    "Answer using only the context below. Be direct, clear, and grounded.\n"
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

print("MAAT Knowledge System ready.")
print("Type your question and press Enter. Type quit to exit.")
print("=" * 50)

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        print("\nPhore You.")
        break
    if not question:
        continue
    answer = chain.invoke(question)
    print(f"\nMAAT: {answer}")
    print("-" * 50)
