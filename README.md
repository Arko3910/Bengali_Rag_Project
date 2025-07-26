
# 📘 Bangla-English RAG Chatbot (mT5)

This project is a Bangla-English Retrieval-Augmented Generation (RAG) system designed to answer questions from uploaded Bangla or English PDF documents using HuggingFace's multilingual models.

---

## 🚀 Features

- 📤 Upload Bangla or English PDFs
- 🔎 Ask questions in Bangla or English
- 📚 Uses vector similarity search to retrieve context
- 🤖 Generates context-aware answers using mT5 LLM
- 🛡️ Handles fallback and document parsing errors
- 🌐 Streamlit-powered UI

---

## 🛠️ Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment Variable

Set your Hugging Face API key in a `.env` file or your system environment:

```bash
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🔍 Sample Bangla Queries

```text
অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
```

---

## ✅ Answers to Assignment Questions

### 1. **What method or library did you use to extract the text, and why?**

We used **PyMuPDFLoader** from `langchain_community.document_loaders`. It provides reliable text extraction from PDFs, including basic Bangla support.  
Yes, we faced **formatting challenges** like garbled OCR text and missing characters in Bangla. So we added **cleaning filters** to normalize punctuation and remove noise.

---

### 2. **What chunking strategy did you choose? Why?**

We used `RecursiveCharacterTextSplitter` with:
- `chunk_size = 1000`
- `chunk_overlap = 200`
- Custom separators (`["\n\n", "\n", "।", ".", "!", "?", " "]`)

This hybrid strategy is ideal for **semantic retrieval** in multilingual documents. Overlap preserves context and avoids semantic breaks.

---

### 3. **What embedding model did you use? Why?**

We used `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.  
It supports **Bangla + English**, is efficient, and works well with FAISS for dense retrieval. It captures contextual and semantic meaning across languages.

---

### 4. **How are you comparing the query with your stored chunks?**

We use **FAISS** for approximate nearest neighbor search based on cosine similarity.  
This method is fast, scalable, and works well with high-dimensional embeddings.

---

### 5. **How do you ensure meaningful comparison between query and documents?**

- We detect language (Bangla/English) and adjust prompts accordingly.
- We clean and validate document chunks before vectorization.
- mT5 is used for Bangla-aware answering.
- If results are vague, we apply **fallback sentence matching** from source text.

If the query is vague, we default to showing **most relevant chunks** or fallback to **first coherent sentences**.

---

### 6. **Do the results seem relevant? What could improve them?**

Yes, results are mostly relevant for structured or clear PDFs.  
Improvements could include:
- Advanced OCR (e.g. `pytesseract`) for scanned PDFs
- Improved chunk filtering logic
- Use of `ragas` or `bertopic` for evaluation
- Integrating larger multilingual LLMs (e.g., `mistralai/Mixtral`)

---

## 📎 Tools & Stack

| Component      | Tool/Library                                 |
|----------------|----------------------------------------------|
| LLM            | `google/mt5-base` (via HuggingFace)          |
| Embeddings     | `sentence-transformers/paraphrase-mpnet`     |
| Vector DB      | FAISS                                        |
| Interface      | Streamlit                                    |
| Chunking       | LangChain's RecursiveCharacterTextSplitter   |
| Extraction     | LangChain's PyMuPDFLoader                    |

---

## 🧪 Evaluation

Evaluation was done manually using sample Bangla questions.  
Groundedness and relevance were verified through source text comparison.

---

## 📫 Contact

For feedback: **asifuzzamanarko@gmail.com**

---
