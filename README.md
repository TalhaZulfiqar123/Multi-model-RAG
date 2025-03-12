**Multi-Model RAG Chatbot**

### **Overview:**
This project is a **Retrieval-Augmented Generation (RAG) chatbot** built using **Streamlit** and **LangChain**. It allows users to interact with multiple AI models and retrieve information from uploaded PDF documents.

### **Features:**
- **Multi-Model Support**: Compare responses from different AI models including:
  - `llama3-8b-8192`
  - `mixtral-8x7b-32768`
  - `qwen-2.5-32b`
- **PDF-Based Retrieval**: Upload PDFs to provide contextual responses based on document content.
- **Streamlit UI**: Interactive chat interface with model selection.
- **LangChain Integration**: Uses LangChain components for embedding, retrieval, and querying.

### **Usage:**
1. **Run the Streamlit app:**
   ```
   streamlit run app.py
   ```
2. **Upload PDF documents** for retrieval-augmented responses.
3. **Enter a prompt** in the chat input field and compare responses from different models.

### **File Structure:**
- **`app.py`**: Main Streamlit application
- **`requirements.txt`**: Required dependencies
- **`README.md`**: Project documentation
- **`.env`**: Environment variables (optional)

### **Dependencies:**
- `streamlit`
- `langchain`
- `langchain_groq`
- `PyPDFLoader`
- `HuggingFaceEmbeddings`
- `os`

### **Future Improvements:**
- **Add support for more LLM providers** (OpenAI, Anthropic, etc.).
- **Improve document retrieval efficiency.**
- **Implement a database for storing chat history.**


