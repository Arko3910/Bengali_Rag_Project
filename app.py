import gradio as gr
import os
import tempfile
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint


# -----------------------------
# MODEL SETTINGS
# -----------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 100 percent working text-generation model
LLM_MODEL = "tiiuae/falcon-7b-instruct"


def load_embedder():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )


# -----------------------------
# PROCESS PDF
# -----------------------------
def process_pdf(pdf_file):
    """Convert uploaded PDF to chunks + FAISS index."""
    try:
        # Gradio provides a file path directly
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else pdf_file

        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()

        cleaned_docs = []
        for d in docs:
            text = re.sub(r"\s+", " ", d.page_content)
            if len(text) > 20:
                d.page_content = text
                cleaned_docs.append(d)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(cleaned_docs)

        embedder = load_embedder()
        vector_store = FAISS.from_documents(chunks, embedder)

        return vector_store, f"PDF processed successfully. Total chunks: {len(chunks)}"

    except Exception as e:
        return None, f"Error: {e}"


# -----------------------------
# LOAD LLM
# -----------------------------
def load_llm():
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        return None

    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        temperature=0.2,
        max_new_tokens=400,
        huggingfacehub_api_token=token,
    )


# -----------------------------
# RAG CHAT FUNCTION
# -----------------------------
def rag_chat(message, history, vector_store):
    if vector_store is None:
        return history + [[{"role": "user", "content": message},
                           {"role": "assistant", "content": "Please upload and process a PDF first."}]]

    try:
        docs = vector_store.similarity_search(message, k=4)
        context = "\n\n".join([d.page_content for d in docs])

        llm = load_llm()
        if llm is None:
            return history + [[{"role": "user", "content": message},
                               {"role": "assistant", "content": "Missing HF API token."}]]

        prompt = f"""
Answer ONLY using the context below.
If the answer is not in the document, say:
"The document does not contain this information."

CONTEXT:
{context}

QUESTION:
{message}

ANSWER:
"""

        # CORRECT WAY FOR HuggingFaceEndpoint:
        response = llm.invoke(prompt)

        answer = str(response).strip()

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        return history

    except Exception as e:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history




# -----------------------------
# DARK THEME CSS
# -----------------------------
DARK_CSS = """
body, .gradio-container {
    background-color: #0D0D0D !important;
    color: white !important;
}
.chatbot {
    background-color: #1A1A1A !important;
}
.button-primary {
    background: #00A6FF !important;
    color: white !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    padding: 12px;
}
"""


# -----------------------------
# BUILD UI
# -----------------------------
def build_ui():
    with gr.Blocks(title="RAG Chatbot") as ui:

        gr.HTML(f"<style>{DARK_CSS}</style>")

        gr.Markdown(
            """
            <h1 style="text-align:center;color:#00A6FF;font-weight:700;">RAG Chatbot</h1>
            <p style="text-align:center;color:#CCCCCC;">
                Upload a PDF → Ask questions → Get accurate answers directly from the document.
            </p>
            """
        )

        vector_store_state = gr.State(None)

        with gr.Row():

            # LEFT SIDE — Upload PDF
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                status_box = gr.Textbox(label="Status", interactive=False)
                process_btn = gr.Button("Process PDF", elem_classes="button-primary")

                process_btn.click(
                    fn=process_pdf,
                    inputs=[pdf_input],
                    outputs=[vector_store_state, status_box]
                )

            # RIGHT SIDE — Chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat with PDF", height=500)

                user_msg = gr.Textbox(
                    label="Ask a Question",
                    placeholder="Type your question about the PDF..."
                )
                send_btn = gr.Button("Send", elem_classes="button-primary")

                send_btn.click(
                    fn=rag_chat,
                    inputs=[user_msg, chatbot, vector_store_state],
                    outputs=[chatbot]
                )

        return ui


app = build_ui()


if __name__ == "__main__":
    app.launch(server_port=7860, ssr_mode=False)
