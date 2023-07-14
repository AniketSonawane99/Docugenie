
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
# Bring in streamlit for UI/app interface
import streamlit as st
from langchain.vectorstores import Chroma
import PyPDF2

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
st.title("Document Genie: TALK TO YOUR PDF :page_facing_up:")
st.info("""
        If there are multiple PDF's, all the information from them is gathered as one big database, so the summary function may not be accurate, sorry!
        
        
        Made by Aniket Sonawane with inspiration and help from various sources such as Sam Whiteeven and others
        """)
uploaded_files = st.file_uploader("Put your document here", type ="pdf", accept_multiple_files= True)
if st.button("Process"):
    st.session_state.text_list = []
    for file in uploaded_files:
        pdfReader = PyPDF2.PdfReader(file)
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          st.session_state.text_list.append(text)
    embeddings = OpenAIEmbeddings(openai_api_key= st.secrets["openai_api_key"])
    st.session_state.store = Chroma.from_texts(st.session_state.text_list,embedding=embeddings)
    st.session_state.retriever  = st.session_state.store.as_retriever(search_kwargs={"k": len(uploaded_files)})
    st.session_state.llm = OpenAI(temperature=0, openai_api_key= st.secrets["openai_api_key"])
    chain = load_summarize_chain(st.session_state.llm, chain_type="stuff")
    search = st.session_state.store.similarity_search(" ")
    st.session_state.summary = chain.run(input_documents=search, question="Write a summary within 200 words.")

if "summary" in st.session_state:
    st.write(st.session_state.summary)

prompt = st.text_input('Ask questions about the pdf to the pdf')
if st.button("Answer") or prompt:
    qa_chain = RetrievalQA.from_chain_type(llm = st.session_state.llm,
                                            chain_type= "stuff",
                                            retriever = st.session_state.retriever)
    result = qa_chain(prompt)
    st.success(result['result'])
