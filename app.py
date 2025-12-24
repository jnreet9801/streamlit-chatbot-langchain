import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="📄 Document Q&A", layout="centered")

st.title("📄 Document Q&A Chatbot")
st.write("Upload a PDF and ask questions from it.")


# -------------------- Upload PDF --------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


# -------------------- Process PDF --------------------
if uploaded_file is not None:
    with st.spinner("Processing document..."):

        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)

        # Embeddings (FREE)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # LLM (FREE)
        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation",
            model_kwargs={"temperature": 0.0, "max_length": 512}
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """Answer the question using ONLY the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {question}
            """
        )

        # RAG Chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    st.success("✅ Document processed successfully!")


    # -------------------- Ask Question --------------------
    question = st.text_input("Ask a question from the document")

    if question:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)

        st.markdown("### ✅ Answer")
        st.write(answer)
