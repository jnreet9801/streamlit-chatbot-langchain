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


st.set_page_config(page_title="📄 Document Q&A", layout="centered")

st.title("📄 Document Q&A Chatbot")
st.write("Upload a PDF and ask questions from it.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])


if uploaded_file is not None:
    with st.spinner("Processing document..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-small",
            task="text2text-generation",
            model_kwargs={"temperature": 0.0, "max_length": 512}
        )

        prompt = ChatPromptTemplate.from_template(
            """You are an expert assistant.
                Using ONLY the information from the context below:
                - Answer the question directly and accurately
                - Keep the answer short, clear, and to the point
                - Do not add extra information
                - Do not make assumptions
                
                If the answer is not explicitly present in the context, reply exactly:
                "I don't know."

            Context:
            {context}

            Question:
            {question}
            """
        )

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    st.success("✅ Document processed successfully!")


    question = st.text_input("Ask a question from the document")

    if question:
        with st.spinner("Thinking..."):
            answer = rag_chain.invoke(question)

        st.markdown("### ✅ Answer")
        st.write(answer)


