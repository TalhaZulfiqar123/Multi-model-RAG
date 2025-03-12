import streamlit as st  
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 

# Imports for Retrieval-Augmented Generation (RAG)
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Set Streamlit app title
st.title('Multi-Model RAG Chatbot!')

# Sidebar for model selection
st.sidebar.header("Model Selection")
available_models = ["llama3-8b-8192", "mixtral-8x7b-32768", "qwen-2.5-32b"]
selected_models = st.sidebar.multiselect("Choose models to compare:", available_models, default=["llama3-8b-8192"])

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Function to process PDFs into a vector store
@st.cache_resource
def get_vectorstore(pdf_paths):
    if not pdf_paths:
        return None
    try:
        loaders = [PyPDFLoader(pdf) for pdf in pdf_paths]
        index = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        ).from_loaders(loaders)
        return index.vectorstore
    except Exception as e:
        st.error(f"Error loading PDFs: {str(e)}")
        return None

# File uploader
if not st.session_state.pdf_paths:
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        pdf_paths = []
        for uploaded_file in uploaded_files:
            pdf_path = f"./temp_{uploaded_file.name}"
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_paths.append(pdf_path)
        st.session_state.pdf_paths = pdf_paths
        st.session_state.vectorstore = get_vectorstore(pdf_paths)
        st.success("PDFs uploaded and processed successfully!")

# User input
prompt = st.chat_input('Pass your prompt here')
if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Loop through selected models
    responses = {}
    for model in selected_models:
        groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
        )
        
        try:
            vectorstore = st.session_state.vectorstore
            if vectorstore is None:
                response = f"[{model}] No PDF uploaded. Answering without additional context."
            else:
                # Debugging: Check if vectorstore is correctly created
                if vectorstore is None:
                    st.error("Vector store is empty. Ensure PDFs are uploaded and processed.")

                chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type='stuff',
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True
                )

                # Custom prompt to ensure answers are based only on provided content
                custom_prompt = f"""
                Answer the following query strictly based on the provided content.
                If there is no relevant information in the provided content, respond with:
                "No content available."

                Query: {prompt}
                """

                result = chain({"query": custom_prompt})
                response = result.get("result", "No content available.")

            responses[model] = response
        
        except Exception as e:
            responses[model] = f"Error: {str(e)}"

    # Display model responses
    for model, response in responses.items():
        st.subheader(f"Response from {model}:")
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': f'[{model}] {response}'})
