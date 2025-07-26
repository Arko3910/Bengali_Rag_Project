import os
import tempfile
import traceback
import re
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
# Make sure to set your API token as environment variable for security
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Better LLM models that are more likely to work
LLM_MODELS = [
    "google/flan-t5-base",
    "microsoft/DialoGPT-medium", 
    "huggingface/CodeBERTa-small-v1",
    "distilbert-base-uncased"
]

# Better prompts optimized for mT5
BANGLA_PROMPT = """প্রশ্ন: {question}

নিম্নলিখিত তথ্যের ভিত্তিতে উত্তর দিন:

{context}

উত্তর:"""

ENGLISH_PROMPT = """Question: {question}

Based on the following context, provide a clear answer:

{context}

Answer:"""

def detect_language(text):
    """Detect if text is primarily Bangla or English"""
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    return 'bangla' if bangla_chars > english_chars else 'english'

def clean_text_output(text):
    """Clean up the generated text and fix encoding issues"""
    if not text:
        return "No answer found."
    
    # Convert to string and strip
    text = str(text).strip()
    
    # Remove common model artifacts and HTML tags
    text = re.sub(r'^(Answer:|উত্তর:)', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'<[^>]*>', '', text)  # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets
    text = re.sub(r'\{.*?\}', '', text)  # Remove curly brackets
    
    # Fix common OCR/encoding issues with Bangla text
    text = re.sub(r'([।\.!?])\s*([।\.!?])', r'\1', text)  # Remove duplicate punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Check if text looks corrupted (no spaces, too many punctuation marks)
    word_count = len(text.split())
    char_count = len(text)
    if word_count > 0 and char_count / word_count > 20:  # Average word length too high
        return "The text appears to be corrupted. Please try with a clearer PDF document."
    
    # Split into sentences properly
    # Use both English and Bangla sentence endings
    sentences = re.split(r'([।.!?])', text)
    cleaned_sentences = []
    
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            sentence = sentences[i].strip() + sentences[i+1]
            if sentence.strip() and len(sentence.strip()) > 3:
                # Check if sentence has reasonable structure
                if ' ' in sentence or re.search(r'[\u0980-\u09FF]', sentence):
                    cleaned_sentences.append(sentence.strip())
    
    result = '\n\n'.join(cleaned_sentences) if cleaned_sentences else text
    
    # Final check for corrupted text
    if not result.strip() or len(result.replace(' ', '')) < 10:
        return "Unable to generate a clear answer from the document."
    
    return result

# --- Cache functions ---
@st.cache_resource
def load_embeddings():
    """Load embeddings model"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        return embeddings
    except Exception as e:
        try:
            # Fallback to a smaller model
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"}
            )
            return embeddings
        except Exception as e2:
            st.error(f"Failed to load embedding models: {e2}")
            return None

@st.cache_resource
def load_llm():
    """Load LLM with focus on Bangla-capable models"""
    
    # Validate API token first
    if not HUGGINGFACEHUB_API_TOKEN or HUGGINGFACEHUB_API_TOKEN == "":
        return create_fallback_llm()
    
    # Try these models in order of preference, prioritizing Bangla support
    models_to_try = [
        "google/mt5-base",  # Best for multilingual including Bangla
        "google/flan-t5-base",
        "google/mt5-small",
        "google/flan-t5-small"
    ]
    
    for model in models_to_try:
        try:
            # Fixed parameters for mT5 and T5 models
            llm = HuggingFaceEndpoint(
                repo_id=model,
                temperature=0.2,  # Lower for more consistent outputs
                max_new_tokens=200,  # Increased for better Bangla responses
                huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
            )
            
            # Test the model with a simple Bangla query
            if "mt5" in model.lower():
                test_response = llm.invoke("বাংলাদেশের রাজধানী কী?")
            else:
                test_response = llm.invoke("What is the capital of Bangladesh?")
            
            if test_response and len(str(test_response).strip()) > 3:
                return llm
                
        except Exception as e:
            continue
    
    # If all models fail, create fallback
    return create_fallback_llm()

def create_fallback_llm():
    """Create a more intelligent fallback that provides better Bangla answers"""
    class FallbackLLM:
        def invoke(self, prompt):
            try:
                # Extract context and question more carefully
                lines = [line.strip() for line in prompt.split('\n') if line.strip()]
                context_lines = []
                question_line = ""
                language = 'english'
                
                # Find the question and detect language
                for line in lines:
                    if line.startswith(('Question:', 'প্রশ্ন:')):
                        question_line = line.replace('Question:', '').replace('প্রশ্ন:', '').strip()
                        if 'প্রশ্ন:' in line:
                            language = 'bangla'
                        break
                
                # Extract context more accurately
                in_context_section = False
                for line in lines:
                    if ('following information:' in line.lower() or 
                        'নিচের তথ্য থেকে' in line.lower()):
                        in_context_section = True
                        continue
                    elif line.startswith(('Answer:', 'উত্তর:')):
                        break
                    elif in_context_section and line and not line.startswith(('Question:', 'প্রশ্ন:')):
                        context_lines.append(line)
                
                if not context_lines:
                    return "কোন প্রাসঙ্গিক তথ্য পাওয়া যায়নি।" if language == 'bangla' else "No relevant information found."
                
                # Join context and clean it
                context_text = ' '.join(context_lines)
                
                # Clean context text - remove garbled characters
                context_text = re.sub(r'[^\u0980-\u09FF\s\w.,!?।]', ' ', context_text)  # Keep Bangla, English, and basic punctuation
                context_text = re.sub(r'\s+', ' ', context_text).strip()
                
                # Split into meaningful sentences
                if language == 'bangla':
                    sentences = [s.strip() for s in context_text.split('।') if s.strip() and len(s.strip()) > 10]
                else:
                    sentences = [s.strip() for s in context_text.split('.') if s.strip() and len(s.strip()) > 10]
                
                if not sentences:
                    return "নথিতে পাঠযোগ্য তথ্য পাওয়া যায়নি।" if language == 'bangla' else "No readable information found in the document."
                
                # Simple keyword matching for better responses
                question_lower = question_line.lower()
                
                # Find the most relevant sentences
                relevant_sentences = []
                question_words = [w for w in question_lower.split() if len(w) > 2]
                
                for sentence in sentences[:10]:  # Check first 10 sentences
                    sentence_lower = sentence.lower()
                    score = 0
                    
                    # Calculate relevance score
                    for word in question_words:
                        if word in sentence_lower:
                            score += 1
                    
                    if score > 0:
                        relevant_sentences.append((sentence, score))
                
                # Sort by relevance and take top sentences
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                
                if relevant_sentences:
                    # Take top 2-3 most relevant sentences
                    top_sentences = [s[0] for s in relevant_sentences[:3]]
                    result = ('।' if language == 'bangla' else '.').join(top_sentences)
                    if language == 'bangla':
                        result += '।'
                    else:
                        result += '.'
                    return result
                
                # If no keyword match, return first few coherent sentences
                coherent_sentences = [s for s in sentences[:3] if len(s) > 20]
                if coherent_sentences:
                    result = ('।' if language == 'bangla' else '.').join(coherent_sentences[:2])
                    if language == 'bangla':
                        result += '।'
                    else:
                        result += '.'
                    return result
                
                return "উপযুক্ত উত্তর খুঁজে পাওয়া যায়নি।" if language == 'bangla' else "Could not find a suitable answer."
                
            except Exception as e:
                return f"প্রশ্ন প্রক্রিয়াকরণে সমস্যা হয়েছে।" if 'bangla' in str(e).lower() else f"Error processing question."
        
        def __call__(self, prompt):
            return self.invoke(prompt)
    
    return FallbackLLM()

def create_qa_system(vector_store, llm):
    """Create a robust QA system with better error handling"""
    
    def answer_question(query):
        try:
            # Detect language
            language = detect_language(query)
            
            # Get relevant documents
            docs = vector_store.similarity_search(query, k=4)
            if not docs:
                return {
                    "result": "No relevant information found in the document.",
                    "source_documents": [],
                    "language": language
                }
            
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Choose prompt based on language
            if language == 'bangla':
                prompt_template = BANGLA_PROMPT
            else:
                prompt_template = ENGLISH_PROMPT
            
            # Create the prompt with better formatting for mT5
            full_prompt = prompt_template.format(
                question=query,
                context=context[:1800]  # Increased context for mT5 which handles longer text better
            )
            
            # Debug: Show the prompt being used
            if st.session_state.get('debug_mode', False):
                st.write("**Debug - Full Prompt:**")
                st.text(full_prompt)
            
            # Get response from LLM with better handling for mT5
            try:
                if hasattr(llm, 'invoke'):
                    response = llm.invoke(full_prompt)
                else:
                    response = llm(full_prompt)
                
                # Debug: Show raw response
                if st.session_state.get('debug_mode', False):
                    st.write("**Debug - Raw Response:**")
                    st.text(str(response))
                
                # Better post-processing for mT5 responses
                response_str = str(response).strip()
                
                # mT5 sometimes returns the full prompt + answer, extract just the answer
                if "উত্তর:" in response_str:
                    response_str = response_str.split("উত্তর:")[-1].strip()
                elif "Answer:" in response_str:
                    response_str = response_str.split("Answer:")[-1].strip()
                
                response = response_str
                
            except Exception as llm_error:
                st.warning(f"LLM error: {str(llm_error)}")
                # Create a simple context-based response
                sentences = context.split('।' if language == 'bangla' else '.')[:3]
                fallback_msg = "Based on the document:" if language == 'english' else "নথি অনুযায়ী:"
                response = f"{fallback_msg}\n\n{('।' if language == 'bangla' else '.').join(sentences)}{'।' if language == 'bangla' else '.'}"
            
            # Clean the response
            cleaned_response = clean_text_output(str(response))
            
            # If cleaned response is too short or generic, provide context directly
            if (len(cleaned_response) < 30 or 
                "unable to" in cleaned_response.lower() or 
                "no answer" in cleaned_response.lower() or
                "corrupted" in cleaned_response.lower()):
                
                # Extract most relevant sentences from context
                context_sentences = [s.strip() for s in context.replace('।', '.').split('.') if s.strip()]
                relevant_sentences = []
                
                # Simple keyword matching with better filtering
                query_words = [word.lower() for word in query.split() if len(word) > 2]
                
                for sentence in context_sentences[:8]:  # Check more sentences
                    if len(sentence) > 30:  # Skip very short sentences
                        sentence_lower = sentence.lower()
                        # Check if sentence contains query keywords
                        score = sum(1 for word in query_words if word in sentence_lower)
                        if score > 0:
                            relevant_sentences.append((sentence, score))
                
                # Sort by relevance score and take top sentences
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                
                if relevant_sentences:
                    top_sentences = [s[0] for s in relevant_sentences[:3]]
                    fallback_msg = "Based on the document:" if language == 'english' else "নথি অনুযায়ী:"
                    cleaned_response = f"{fallback_msg}\n\n{'. '.join(top_sentences)}."
                else:
                    # Last resort: use first coherent sentences
                    coherent_sentences = [s for s in context_sentences[:5] if len(s) > 30 and ' ' in s]
                    if coherent_sentences:
                        fallback_msg = "From the document:" if language == 'english' else "নথি থেকে:"
                        cleaned_response = f"{fallback_msg}\n\n{'. '.join(coherent_sentences[:2])}."
                    else:
                        cleaned_response = "The document text appears to be corrupted or unreadable. Please try uploading a clearer PDF."
            
            return {
                "result": cleaned_response,
                "source_documents": docs,
                "language": language
            }
            
        except Exception as e:
            st.error(f"Error in QA system: {str(e)}")
            # Last resort fallback
            try:
                docs = vector_store.similarity_search(query, k=2)
                context = "\n\n".join([doc.page_content[:400] for doc in docs])
                
                fallback_msg = "Based on the document:" if detect_language(query) == 'english' else "নথি অনুযায়ী:"
                
                return {
                    "result": f"{fallback_msg}\n\n{context}",
                    "source_documents": docs,
                    "language": detect_language(query),
                    "fallback": True
                }
            except:
                return {
                    "result": "Sorry, I couldn't process your question. Please try rephrasing it or check if the document was uploaded properly.",
                    "source_documents": [],
                    "language": detect_language(query),
                    "fallback": True
                }
    
    return answer_question

# --- Streamlit App ---
st.set_page_config(page_title="📘 Bangla RAG Chatbot", layout="wide")
st.title("📘 Bangla-English PDF Q&A System")

# Sidebar
with st.sidebar:
    st.header("🔧 System Status")
    
    # API Token status
    if HUGGINGFACEHUB_API_TOKEN and HUGGINGFACEHUB_API_TOKEN != "":
        st.success("🔑 API Token: Set")
    else:
        st.error("🔑 API Token: Missing")
        st.info("Set HUGGINGFACEHUB_API_TOKEN environment variable")
    
    if 'processed_docs' in st.session_state:
        st.success(f"✅ Document processed")
        st.metric("Text Chunks", st.session_state.processed_docs)
    else:
        st.info("📄 No document loaded")
    
    if 'llm_loaded' in st.session_state and st.session_state.llm_loaded:
        st.success("🤖 AI Model Ready (mT5 for Bangla)")
    else:
        st.warning("🤖 AI Model Not Loaded")
        if st.button("🔄 Retry Loading Models"):
            # Clear cache and retry
            st.cache_resource.clear()
            st.rerun()

# File upload
uploaded_file = st.file_uploader("📤 Upload PDF Document", type="pdf")

# Process PDF
if uploaded_file:
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        
        with st.spinner("🔄 Processing document..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load and split document
                loader = PyMuPDFLoader(tmp_path)
                documents = loader.load()
                
                # Check if document loaded properly
                if not documents:
                    st.error("❌ Could not extract text from PDF. Please check if the PDF is readable.")
                    st.stop()
                
                # Better text cleaning before splitting
                cleaned_documents = []
                for doc in documents:
                    # Clean the text content
                    content = doc.page_content
                    
                    # Remove common OCR artifacts
                    content = re.sub(r'[^\u0980-\u09FF\s\w.,!?।\-\(\)]', ' ', content)  # Keep only valid characters
                    content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                    content = re.sub(r'([।.])\s*([।.])', r'\1', content)  # Remove duplicate punctuation
                    
                    doc.page_content = content.strip()
                    if len(doc.page_content) > 30:  # Only keep documents with substantial content
                        cleaned_documents.append(doc)
                
                documents = cleaned_documents
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # Larger chunks for better context
                    chunk_overlap=200,  # More overlap for continuity
                    separators=["\n\n", "\n", "।", ".", "!", "?", " "],
                    length_function=len
                )
                docs = text_splitter.split_documents(documents)
                
                # Better filtering of chunks
                filtered_docs = []
                for doc in docs:
                    content = doc.page_content.strip()
                    
                    # More sophisticated content validation
                    word_count = len(content.split())
                    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', content))
                    english_chars = len(re.findall(r'[a-zA-Z]', content))
                    
                    # Keep chunks that have reasonable structure
                    if (len(content) > 80 and 
                        word_count >= 5 and
                        (bangla_chars > 10 or english_chars > 20)):
                        filtered_docs.append(doc)
                
                docs = filtered_docs
                if not docs:
                    st.error("❌ No readable text chunks found in the PDF.")
                    st.stop()

                # Load models silently
                embeddings = load_embeddings()
                if not embeddings:
                    st.error("❌ Failed to load embedding model")
                    st.stop()

                llm = load_llm()
                if not llm:
                    st.error("❌ Failed to load language model")
                    st.stop()

                # Create vector store
                st.info("🗃️ Creating vector database...")
                vector_store = FAISS.from_documents(docs, embeddings)
                
                # Create QA system
                qa_system = create_qa_system(vector_store, llm)

                # Store in session
                st.session_state.vector_store = vector_store
                st.session_state.qa_system = qa_system
                st.session_state.current_file = uploaded_file.name
                st.session_state.processed_docs = len(docs)
                st.session_state.llm_loaded = True

                # Cleanup
                os.unlink(tmp_path)
                
                st.success(f"✅ Successfully processed {len(docs)} text chunks!")

            except Exception as e:
                st.error("❌ Document processing failed")
                st.error(f"Error: {str(e)}")
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())

# Q&A Section
if 'qa_system' in st.session_state:
    st.subheader("💬 Ask Your Question")
    
    # Example questions
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🇧🇩 Bangla Examples:**")
        st.markdown("- এই গল্পের মূল চরিত্র কে?")
        st.markdown("- কাহিনীর সারাংশ কী?")
        st.markdown("- এই নথিতে কী বলা হয়েছে?")
    
    with col2:
        st.markdown("**🇺🇸 English Examples:**")
        st.markdown("- Who is the main character?")
        st.markdown("- What is the story about?")
        st.markdown("- What does this document say?")
    
    # Question input
    question = st.text_area(
        "📝 Your Question:",
        height=100,
        placeholder="Ask anything about your document in Bangla or English..."
    )
    
    if st.button("🔍 Get Answer", type="primary"):
        if question.strip():
            with st.spinner("🤖 Generating answer..."):
                try:
                    result = st.session_state.qa_system(question)
                    
                    # Display answer
                    st.markdown("### ✅ Answer:")
                    
                    if 'fallback' in result:
                        st.warning("⚠️ Using simple search (AI model had issues)")
                    
                    # Show the answer in a nice box that matches Streamlit's theme
                    st.markdown(f"""
                    <div style="background-color: var(--background-color, #0e1117); 
                                padding: 20px; 
                                border-radius: 10px; 
                                border-left: 4px solid #28a745; 
                                border: 1px solid #262730;
                                margin: 10px 0;">
                        <p style="color: var(--text-color, #fafafa); 
                                 font-size: 16px; 
                                 margin: 0; 
                                 line-height: 1.6; 
                                 font-weight: 500;">{result['result']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources
                    if result['source_documents']:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(result['source_documents'], 1):
                                st.markdown(f"**📄 Source {i}:**")
                                content = doc.page_content[:500]
                                st.text(content + "..." if len(doc.page_content) > 500 else content)
                                if 'page' in doc.metadata:
                                    st.caption(f"Page: {doc.metadata['page']}")
                                st.markdown("---")
                    
                    # Language info
                    lang_name = "Bangla" if result['language'] == 'bangla' else "English"
                    st.caption(f"🌐 Detected Language: {lang_name}")
                    
                except Exception as e:
                    st.error("❌ Failed to generate answer")
                    st.error(f"Error: {str(e)}")
                    with st.expander("🔍 Debug Info"):
                        st.code(traceback.format_exc())
                    
        else:
            st.warning("⚠️ Please enter a question!")

else:
    st.info("👆 Upload a PDF document to get started!")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Steps:
        1. **Setup**: Make sure you have a valid HuggingFace API token
        2. **Upload**: Select your PDF file (Bangla or English content)
        3. **Wait**: System will process and index the document
        4. **Ask**: Type your question in Bangla or English
        5. **Get Answer**: Receive formatted answer with source references
        
        ### Features:
        - ✅ Supports both Bangla and English
        - ✅ Automatic language detection
        - ✅ Clean, readable answers
        - ✅ Source document references
        - ✅ Fallback system for reliability
        - ✅ Better error handling
        
        ### Troubleshooting:
        - If models fail to load, try the "Retry Loading Models" button
        - Make sure your HuggingFace API token is valid
        - Check your internet connection
        """)

# Debug section for development
if st.checkbox("🔧 Debug Mode"):
    st.session_state['debug_mode'] = True
    st.markdown("### Debug Information")
    st.write("Session State Keys:", list(st.session_state.keys()))
    if 'llm_loaded' in st.session_state:
        st.write("LLM Loaded:", st.session_state.llm_loaded)
    if 'processed_docs' in st.session_state:
        st.write("Number of document chunks:", st.session_state.processed_docs)
    
    # Test vector search if available
    if 'vector_store' in st.session_state:
        st.subheader("🔍 Test Document Search")
        test_query = st.text_input("Test search query:", placeholder="Try searching for keywords from your document")
        if test_query:
            try:
                test_docs = st.session_state.vector_store.similarity_search(test_query, k=3)
                st.write(f"**Found {len(test_docs)} results:**")
                for i, doc in enumerate(test_docs):
                    st.write(f"**Result {i+1} (Score: similarity):**")
                    content_preview = doc.page_content[:300]
                    st.text(content_preview + "..." if len(doc.page_content) > 300 else content_preview)
                    st.write("---")
                
                if not test_docs:
                    st.warning("No results found. This might indicate:")
                    st.write("1. The document text is corrupted")
                    st.write("2. The search terms don't match document content")
                    st.write("3. The PDF processing failed")
            except Exception as e:
                st.error(f"Search error: {e}")
    
    # Show document chunks info
    if 'processed_docs' in st.session_state:
        st.write("**Document Processing Info:**")
        st.write(f"- Number of text chunks: {st.session_state.processed_docs}")
        st.write(f"- Document loaded: {st.session_state.get('current_file', 'None')}")
        
        # Show a sample of processed text
        if 'vector_store' in st.session_state:
            try:
                sample_docs = st.session_state.vector_store.similarity_search("text", k=1)
                if sample_docs:
                    st.write("**Sample processed text:**")
                    st.text(sample_docs[0].page_content[:200] + "...")
                else:
                    st.warning("No text found in vector store!")
            except:
                st.warning("Could not retrieve sample text")
else:
    st.session_state['debug_mode'] = False

st.markdown("---")
st.markdown("💡 **Tip**: Be specific in your questions for better results!")

