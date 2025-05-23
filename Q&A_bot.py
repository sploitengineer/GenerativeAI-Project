import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import tempfile
import os

st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("Document Q&A Chatbot")

##File Uploader
uploaded_file = st.file_uploader("Upload File", type="pdf", accept_multiple_files=True)

##storing agent is session (to persist after reload)
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

##loading and Uploading to vectorDB
if uploaded_file and st.session_state.qa_chain is None:
    with st.spinner("Processing the doc and building vector store..."):
        all_texts = []
        for uploaded_file in uploaded_file:
            ##saving pdf to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            ##load pdf and splitting it into chunks
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap= 200)
            chunks = splitter.split_documents(docs)
            all_texts.extend(chunks)

            os.remove(tmp_path)
        
        ##creating vector embedding using huggingface model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        ##creating FAISS vector store
        vector_store = FAISS.from_documents(all_texts, embeddings)

        ##Initialize Llama
        llm = Ollama(model="llama3:8b")

        ##prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful AI assistant. Use the following context to answer the question."
            ),
            HumanMessagePromptTemplate.from_template(
                "Context:\n{context}\n---\nQuestion: {input}"
            ),
        ])

        ##creating retrivalQA chain with LLM and vector store retriever
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt
        )

        qa_chain = create_retrieval_chain(
            retriever = vector_store.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain= combine_docs_chain
        )

        st.session_state.qa_chain = qa_chain

##chat interface
if st.session_state.qa_chain:
    question = st.text_input("Ask question from the uploaded document:")
    if question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.invoke({"input": question})
            st.markdown("### Answer:")
            st.write(response["answer"])