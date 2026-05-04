from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

KNOWLEDGE_BASE_PATH = "/Users/sidart/moor-ai-systems/maat_knowledge_base"
embeddings = OpenAIEmbeddings()

print("Loading knowledge base...")
vectorstore = FAISS.load_local(KNOWLEDGE_BASE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Knowledge base loaded.")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a MAAT advisor for Scrolls of MAAT, a legacy wealth platform "
     "built on Kemetic principles and ancestral architecture. "
     "Answer using only the context below. Be direct and grounded. "
     "Always speak as if addressing someone ready to build generational wealth. "
     "If the answer is not in the context, say: That knowledge is not in this scroll yet.\n\n"
     "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

chat_history = []

def chat(question):
    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))

    return response

print("\nMAAT Advisor ready. I remember our full conversation.")
print("Type your question and press Enter. Type quit to exit.")
print("=" * 50)

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ["quit", "exit", "q"]:
        print("\nPhore You.")
        break
    if not question:
        continue
    answer = chat(question)
    print(f"\nMAAT: {answer}")
    print("-" * 50)
