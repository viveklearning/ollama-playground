from fastapi import FastAPI, File, UploadFile, Form
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "ollama-playground/chat-with-pdf/pdfs"
try:
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vector_store = InMemoryVectorStore(embeddings)
    model = OllamaLLM(model="deepseek-r1:1.5b")
except Exception as e:
    print(f"Error initializing embeddings or model: {e}")

TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
@app.get("/")
async def root():
    return {"message": "PDF Chatbot API is running!"}
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print(f"Saving file to {file_path}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    print("File saved successfully")

    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)
    vector_store.add_documents(chunked_documents)

    return {"message": "PDF uploaded and indexed successfully!"}

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    related_docs = vector_store.similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in related_docs])
    
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    print(response)
    return {"answer": response}
