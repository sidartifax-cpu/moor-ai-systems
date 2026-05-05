import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import glob

load_dotenv()

print("Building FULL MAAT Knowledge Base...")
all_docs = []

pdf_files = glob.glob("/Users/sidart/moor-ai-systems/01-rag-basics/*.pdf")
print(f"Found {len(pdf_files)} PDFs")

for pdf_path in pdf_files:
    try:
        filename = os.path.basename(pdf_path)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)
        print(f"  {filename}: {len(docs)} pages")
    except Exception as e:
        print(f"  Skipped {filename}: {e}")

loader3 = TextLoader("/Users/sidart/moor-ai-systems/maat_website_content.txt", encoding="utf-8")
docs3 = loader3.load()
for doc in docs3:
    doc.metadata["source"] = "scrollsofmaat.com"
all_docs.extend(docs3)
print(f"Website: {len(docs3)} documents")

loader4 = TextLoader("/Users/sidart/moor-ai-systems/maat_membership_info.txt", encoding="utf-8")
docs4 = loader4.load()
for doc in docs4:
    doc.metadata["source"] = "MAAT Membership Info"
all_docs.extend(docs4)
print(f"Membership: {len(docs4)} documents")

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

print("Building vector store...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("/Users/sidart/moor-ai-systems/maat_knowledge_base")
print("Full knowledge base saved.")
