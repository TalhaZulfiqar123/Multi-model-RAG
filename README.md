<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Multi-Model RAG Chatbot</h1>
    
    <h2>Overview</h2>
    <p>This project is a <strong>Retrieval-Augmented Generation (RAG) chatbot</strong> built using <strong>Streamlit</strong> and <strong>LangChain</strong>. The chatbot allows users to interact with multiple AI models and retrieve information from uploaded PDF documents.</p>
    
    <h2>Features</h2>
    <ul>
        <li><strong>Multi-Model Support</strong>: Compare responses from different AI models including:
            <ul>
                <li><code>llama3-8b-8192</code></li>
                <li><code>mixtral-8x7b-32768</code></li>
                <li><code>qwen-2.5-32b</code></li>
            </ul>
        </li>
        <li><strong>PDF-Based Retrieval</strong>: Upload PDFs to provide contextual responses based on document content.</li>
        <li><strong>Streamlit UI</strong>: Interactive chat interface with model selection.</li>
        <li><strong>LangChain Integration</strong>: Utilizes LangChain components for embedding, retrieval, and querying.</li>
    </ul>
    
    <h2>Usage</h2>
    <ol>
        <li>Run the Streamlit app:
            <pre><code>streamlit run phase1.py</code></pre>
        </li>
        <li>Upload PDF documents for retrieval-augmented responses.</li>
        <li>Enter a prompt in the chat input field and compare responses from different models.</li>
    </ol>
    
    <h2>Dependencies</h2>
    <ul>
        <li><code>streamlit</code></li>
        <li><code>langchain</code></li>
        <li><code>langchain_groq</code></li>
        <li><code>PyPDFLoader</code></li>
        <li><code>HuggingFaceEmbeddings</code></li>
        <li><code>os</code></li>
    </ul>
    
    <h2>Future Improvements</h2>
    <ul>
        <li>Add support for more LLM providers (OpenAI, Anthropic, etc.).</li>
        <li>Improve document retrieval efficiency.</li>
        <li>Implement a database for storing chat history.</li>
    </ul>
    <hr>
    <p>Feel free to contribute by creating pull requests or reporting issues!</p>
</body>
</html>

