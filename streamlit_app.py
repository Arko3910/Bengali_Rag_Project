import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA

# --- Configuration ---
HUGGINGFACEHUB_API_TOKEN = "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"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LLM_MODEL = "google/flan-t5-large"

# --- Streamlit UI ---
st.set_page_config(page_title="📘 Bangla RAG Chatbot", layout="wide")
st.title("📘 Bangla-English PDF Q&A Chatbot")

uploaded_file = st.file_uploader("📤 Upload a Bangla PDF", type="pdf")

# --- PDF Upload and Vector Store ---
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("✅ PDF uploaded and processing...")

    # Load and split document
    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(docs, embeddings)

    # LLM
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # --- Q&A Chat ---
    st.subheader("💬 Ask a Question (Bangla or English)")
    user_query = st.text_input("📝 Type your question here...")

    if st.button("🔍 Get Answer") and user_query:
        with st.spinner("🤖 Thinking..."):
            result = qa_chain({"query": user_query})
            st.markdown(f"### ✅ Answer:\n{result['result']}")

            with st.expander("📄 Show Source Document(s)"):
                for doc in result["source_documents"]:
                    st.markdown(doc.page_content[:500] + " ...")

