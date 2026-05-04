from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

app = FastAPI(title="MAAT Advisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE_BASE_PATH = "/app/maat_knowledge_base"
embeddings = OpenAIEmbeddings()

print("Loading MAAT knowledge base...")
vectorstore = FAISS.load_local(KNOWLEDGE_BASE_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Knowledge base ready.")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a MAAT advisor for Scrolls of MAAT, a legacy wealth platform "
     "built on Kemetic principles and ancestral architecture. "
     "Answer using only the context below. Be direct, grounded, and concise. "
     "Always speak as if addressing someone ready to build generational wealth. "
     "If the answer is not in the context, say: That knowledge is not in this scroll yet.\n\n"
     "Context: {context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

sessions = {}

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

class AnswerResponse(BaseModel):
    answer: str
    session_id: str

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

@app.get("/")
def root():
    return {"status": "MAAT Advisor API is live", "version": "1.0"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = []

    chat_history = sessions[request.session_id]
    docs = retriever.invoke(request.question)
    context = format_docs(docs)

    response = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "chat_history": chat_history,
        "question": request.question,
    })

    chat_history.append(HumanMessage(content=request.question))
    chat_history.append(AIMessage(content=response))

    return AnswerResponse(answer=response, session_id=request.session_id)

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "session cleared", "session_id": session_id}
