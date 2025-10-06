import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Load FAISS index
db = FAISS.load_local(
    "C:/Data Science/Azure_Rag/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Initialize LLM
llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)

# Create retrieval chain
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# Streamlit UI
st.title("📄 Azure RAG Chatbot")
st.write("Ask questions about your uploaded PDFs!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your question:")
if query:
    result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
    st.session_state.chat_history.append((query, result["answer"]))

    st.write("### ✅ Answer:")
    st.write(result["answer"])


