import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

os.environ["GOOGLE_API_KEY"] = google_api_key

# Streamlit Page Config
st.set_page_config(
    page_title="Document Question Answering System",
    layout="wide"
)

st.title("📄 Document Question Answering System")
st.write("Upload PDFs inside the Artifacts folder and ask questions from them.")

# Initialize LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Prompt Template
prompt_template = """
Answer the question based only on the provided context.

<context>
{context}
</context>

Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)


# Function to create vector embeddings
def vector_embedding():

    if "vectors" not in st.session_state:

        st.write("Creating embeddings...")

        # Embedding Model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )

        # Load PDFs
        loader = PyPDFDirectoryLoader("./Artifacts")

        docs = loader.load()

        if len(docs) == 0:
            st.error("No PDF files found inside Artifacts folder.")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        final_documents = text_splitter.split_documents(docs)

        # Create FAISS vector store
        vectors = FAISS.from_documents(
            final_documents,
            embeddings
        )

        # Save in session state
        st.session_state.vectors = vectors

        st.success("Vector Store Created Successfully!")


# User input
user_prompt = st.text_input("Ask a Question From Your Documents")


# Button to ingest data
if st.button("Ingest Documents"):

    try:
        vector_embedding()

    except Exception as e:
        st.error(f"Error: {e}")


# Question Answering
if user_prompt:

    try:

        if "vectors" not in st.session_state:
            st.warning("Please ingest documents first.")
        else:

            # Create document chain
            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            # Retriever
            retriever = st.session_state.vectors.as_retriever()

            # Retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            # Response
            start = time.process_time()

            response = retrieval_chain.invoke({
                "input": user_prompt
            })

            end = time.process_time()

            st.write(f"⏱ Response Time: {round(end - start, 2)} seconds")

            st.subheader("Answer")
            st.write(response["answer"])

            # Show retrieved chunks
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")

    except Exception as e:
        st.error(f"Error: {e}")