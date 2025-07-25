import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv()

PDF_PATH = "documents/HSC26-Bangla1st-Paper.pdf"
VECTOR_STORE_PATH = "vector_store/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

def create_vector_store():
    loader = PyMuPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

if not os.path.exists(VECTOR_STORE_PATH):
    vector_store = create_vector_store()
else:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    temperature=0.1,
    max_new_tokens=512,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

app = FastAPI()

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(payload: Query):
    user_query = f"Answer this question based on the provided context:\n{payload.query}"
    result = qa_chain({"query": user_query})
    return {
        "question": payload.query,
        "answer": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bengali RAG API."}
