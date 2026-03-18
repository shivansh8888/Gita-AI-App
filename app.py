import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Gita AI Guide", page_icon="🪷")
st.title("🪷 Bhagavad Gita AI Guide")
st.write("Ask a question, and receive guidance based on the wisdom of the Bhagavad Gita.")

# Securely grab the API key from Streamlit's secret manager
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please set your GOOGLE_API_KEY in the Streamlit secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- 2. BUILD THE AI BRAIN (RAG) ---
# We use @st.cache_resource so we don't rebuild the database every time the user types a letter
@st.cache_resource
def initialize_ai():
    # A. Load the Gita Text
    loader = TextLoader("gita_data.txt")
    docs = loader.load()

    # B. Split text into manageable chunks so the AI can read it easily
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # C. Create Embeddings and Vector Database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant verses

    # D. Setup the Language Model (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # E. Create the System Prompt
    system_prompt = (
        "You are a wise, empathetic spiritual guide. "
        "Use the provided pieces of context from the Bhagavad Gita to answer the user's question. "
        "If you don't know the answer based on the text, say you cannot find the answer in the text. "
        "Keep your answers concise, comforting, and always reference the core message of the verses provided.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # F. Connect the pieces together
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Initialize the chain
rag_chain = initialize_ai()

# --- 3. THE CHAT INTERFACE ---
# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("E.g., How do I deal with anxiety about my exams?"):
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Consulting the Gita..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
            
    # Add AI response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
