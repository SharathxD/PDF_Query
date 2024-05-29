import streamlit as st
#import requests
#import io
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

OPENAI_API_KEY = "api-key"

def process_pdf(file):
    pdfreader = PdfReader(file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search

chain = load_qa_chain(OpenAI(api_key=OPENAI_API_KEY), chain_type="stuff")

st.title("PDF QnA ~ ask anything from the pdf contents <3 ")


uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    
    document_search = process_pdf(uploaded_file)
    st.success("PDF processed successfully.")
    

    user_input = st.text_input("You can shoot the questions now :")
    
    if user_input:
        docs = document_search.similarity_search(user_input)
        response = chain.run(input_documents=docs, question=user_input)
        st.write("Here you go :", response)
