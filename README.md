# Bengali RAG System

This project builds a Retrieval-Augmented Generation (RAG) system that answers questions in Bengali or English based on a provided Bengali PDF.

## Components
- **LLM**: HuggingFace (`FLAN-T5`)
- **Embedding**: Multilingual MPNet
- **Vector DB**: FAISS
- **UI**: Streamlit (demo)

## Setup
```bash
pip install streamlit langchain langchain-community langchain-huggingface sentence-transformers pymupdf faiss-cpu
```
Add your Hugging Face API key in the script or `.env` file.

## Run Streamlit UI
```bash
streamlit run streamlit_app.py
```

## Example Use Case
1. Upload a Bengali PDF (e.g., textbook, story)
2. Ask a question in Bengali or English
3. Get a response from the model based on document content

## Example Query
```bash
অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
```

Expected Answer:
```
শুম্ভুনাথ
```

## 📌 Assignment Questions & Answers

### 1. What method or library did you use to extract the text, and why?
I used `PyMuPDF` to extract text from the Bengali PDF because it supports complex layouts and Unicode text. Formatting issues like page numbers and line breaks were handled with a `clean_text()` function.

### 2. What chunking strategy did you choose?
I used `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=150`. It maintains semantic boundaries while ensuring optimal chunk sizes for embedding.

### 3. What embedding model did you use?
I used `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, which supports both Bengali and English and captures sentence-level semantics effectively.

### 4. How are you comparing the query with your stored chunks?
Using FAISS with cosine similarity between query embeddings and document chunk vectors. FAISS enables fast and scalable retrieval.

### 5. How do you ensure that the question and the document chunks are compared meaningfully?
By using high-quality multilingual embeddings, semantic chunking, and retrieving the top 3 most similar chunks. If the query is vague, the result may be generic or off-topic.

### 6. Do the results seem relevant? If not, what might improve them?
Yes, results are mostly relevant. Improvements could include finer chunking, better embedding models, more contextual documents, or domain-specific tuning.

## License
This project is for educational and research use. Please ensure uploaded PDFs are legally usable.
