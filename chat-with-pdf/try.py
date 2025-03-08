import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define embeddings and vector store
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="deepseek-r1:1.5b")

UPLOAD_DIR = "pdfs"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

TEMPLATE = """
You are an assistant for question-answering tasks. Use the retrieved context to answer concisely.
Question: {question} 
Context: {context} 
Answer:
"""

def process_pdf(file_path):
    """Extracts text from PDF, splits into chunks, and indexes them."""
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    vector_store.add_documents(chunks)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    process_pdf(file_path)
    return {"message": "PDF uploaded and processed successfully!"}

@app.post("/ask/")
async def answer_question(question: str = Form(...)):
    retrieved_docs = vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    if not context:
        return {"answer": "I don't know."}

    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    answer = chain.invoke({"question": question, "context": context})

    return {"answer": answer}
