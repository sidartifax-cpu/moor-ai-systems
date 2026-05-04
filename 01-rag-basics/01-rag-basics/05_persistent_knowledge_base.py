from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

KNOWLEDGE_BASE_PATH = "/Users/sidart/moor-ai-systems/maat_knowledge_base"
embeddings = OpenAIEmbeddings()

if os.path.exists(KNOWLEDGE_BASE_PATH):
    print("Loading saved knowledge base...")
    vectorstore = FAISS.load_local(KNOWLEDGE_BASE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Knowledge base loaded instantly. No API cost.")
else:
    print("Building knowledge base for first time...")
    all_docs = []

    pdf_loader = PyPDFLoader("../MAAT_The_5_Traps.pdf")
    pdf_pages = pdf_loader.load()
    for doc in pdf_pages:
        doc.metadata["source"] = "5 Traps PDF"
    all_docs.extend(pdf_pages)
    print(f"PDF loaded: {len(pdf_pages)} pages")

    text_loader = TextLoader("/Users/sidart/moor-ai-systems/maat_website_content.txt", encoding="utf-8")
    web_docs = text_loader.load()
    for doc in web_docs:
        doc.metadata["source"] = "scrollsofmaat.com"
    all_docs.extend(web_docs)
    print(f"Website loaded: {len(web_docs)} documents")

    membership_loader = TextLoader("/Users/sidart/moor-ai-systems/maat_membership_info.txt", encoding="utf-8")
    membership_docs = membership_loader.load()
    for doc in membership_docs:
        doc.metadata["source"] = "MAAT Membership Info"
    all_docs.extend(membership_docs)
    print(f"Membership info loaded: {len(membership_docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks: {len(chunks)}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(KNOWLEDGE_BASE_PATH)
    print(f"Saved to {KNOWLEDGE_BASE_PATH}")

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template(
    "You are a MAAT advisor for Scrolls of MAAT, a legacy wealth platform "
    "built on Kemetic principles and ancestral architecture.\n"
    "Answer using only the context below. Be direct and grounded.\n"
    "Always speak as if addressing someone ready to build generational wealth.\n"
    "If the answer is not in the context, say: That knowledge is not in this scroll yet.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\nMAAT Advisor ready.")
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
