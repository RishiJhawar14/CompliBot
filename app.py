import os
import re
import json
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, LLMChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(endpoint_url="mistralai/Mixtral-8x7B-Instruct-v0.1",max_new_tokens=256,temperature=0.1,repetition_penalty=1.2)    
llm2 = HuggingFaceEndpoint(endpoint_url="mistralai/Mixtral-8x7B-Instruct-v0.1",max_new_tokens=256,temperature=0.8,repetition_penalty=1.1)    
llm3 = HuggingFaceEndpoint(endpoint_url="mistralai/Mixtral-8x7B-Instruct-v0.1",max_new_tokens=256,temperature=0.001,repetition_penalty=1)

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("./faiss_index")

def get_accessibility_conversational_chain():
    
    accessibility_prompt_template = """

    You are an expert policy regulator that analyze the context deeply and provide violation score on the question. 
    You infer the question in great detail and provide a score between 0 (no violoation) to 100 (high violation) 
    on whether the question violates the terms in the context directly and indirectly. 
    You will only respond with a valid JSON object that has the key Confidence.
    Do not provide explicit explanations.
    
    \n\n

    Context: \n {context} \n
    Question: \n {question} \n

    Answer:"""
    accessibility_prompt = PromptTemplate(template=accessibility_prompt_template, input_variables=["question", "context"])
    accessibility_llm_chain = LLMChain(prompt=accessibility_prompt,llm=llm)

    return accessibility_llm_chain

def get_output_accessibility_conversational_chain():
    
    accessibility_prompt_template = """

    You are an expert policy regulator that analyze the context deeply and provide violation score on the content. 
    You infer the content  and the context in great detail and provide a score between 0 (no violoation) to 100 (high violation) 
    on whether the content violates the context directly and indirectly. 
    You will only respond with a valid JSON object that has the key Confidence. 
    Do not provide explicit explanations.
    
    \n\n

    Context: \n {context} \n
    Content: \n {text} \n   

    Answer:"""
    accessibility_prompt = PromptTemplate(template=accessibility_prompt_template, input_variables=["text", "context"])
    accessibility_llm_chain = LLMChain(prompt=accessibility_prompt,llm=llm2)

    return accessibility_llm_chain

def get_privacy_conversational_chain():
    
    privacy_prompt_template = """

    You are an expert privacy violation detection agent that provide violation score on the question. 
    You infer the question in great detail and provide a score between 0 (no violoation) to 100 (high violation) 
    on whether the question violates the privacy in the context directly or indirectly. 
    You will only respond with a valid JSON object that has the key Confidence. 
    Do not provide explicit explanations.
    
    \n\n

    Context: \n {context} \n
    Question: \n {question} \n

    Answer:"""
    
    privacy_prompt = PromptTemplate(template=privacy_prompt_template, input_variables=["question", "context"])
    privacy_llm_chain = LLMChain(prompt=privacy_prompt,llm=llm)

    return privacy_llm_chain

def get_output_privacy_conversational_chain():
    
    privacy_prompt_template = """

    You are an expert privacy violation detection agent that provide violation score on the content. 
    You infer the content and the context in great detail and provide a score between 0 (no violoation) to 100 (high violation) 
    on whether the content violates the privacy in the context directly or indirectly. 
    You will only respond with a valid JSON object that has the key Confidence. 
    Do not provide explicit explanations.
    
    \n\n

    Context: \n {context} \n
    Content: \n {text} \n

    Answer:"""
    
    privacy_prompt = PromptTemplate(template=privacy_prompt_template, input_variables=["question", "context"])
    privacy_llm_chain = LLMChain(prompt=privacy_prompt,llm=llm2)

    return privacy_llm_chain

def get_sentiment_conversational_chain():
    
    sentiment_prompt_template = """

    You are an expert connection finder (direct or indirect) between the question and context and provide a correlation score. 
    You infer and analyze the question and the context in great depth and provide a score between 0 (no violation) to 100 (high violation).
    You will only respond with a valid JSON object that has the key Confidence.
    Do not provide explicit explanations.
    
    \n\n

    Context: \n {context} \n
    Question: \n {question} \n

    Answer:"""
    
    sentiment_prompt = PromptTemplate(template=sentiment_prompt_template, input_variables=["question", "context"])
    sentiment_llm_chain = LLMChain(prompt=sentiment_prompt,llm=llm)

    return sentiment_llm_chain

def get_output_sentiment_conversational_chain():
    
    sentiment_prompt_template = """

    You are an expert connection finder (direct or indirect) between the content and context and provide a correlation score. 
    You infer and analyze the content and the context in great depth and provide a score between 0 (no violation) to 100 (high violation).
    You will only respond with a JSON object with the key Confidence.
    Do not provide explicit explanations.
    
    \n\n

    Context: \n {context} \n
    Content: \n {text} \n

    Answer:"""
    
    sentiment_prompt = PromptTemplate(template=sentiment_prompt_template, input_variables=["text", "context"])
    sentiment_llm_chain = LLMChain(prompt=sentiment_prompt,llm=llm2)

    return sentiment_llm_chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    accessibility_chain = get_accessibility_conversational_chain()
    privacy_chain = get_privacy_conversational_chain()
    sentiment_chain = get_sentiment_conversational_chain()
    
    accessibility_response = accessibility_chain.invoke(
        {"context":docs, "question": user_question}
        , return_only_outputs=True)
    privacy_response = privacy_chain.invoke(
        {"context":docs, "question": user_question}
        , return_only_outputs=True)
    sentiment_response = sentiment_chain.invoke(
        {"context":docs, "question": user_question}
        , return_only_outputs=True)
    
    # accessibility_response_score = int(json.loads(accessibility_response["text"])["Confidence"])    
    accessibility_response_score = int(re.findall(r'\d+', accessibility_response["text"])[0])
        
    # privacy_response_score = int(json.loads(privacy_response["text"])["Confidence"])
    privacy_response_score = int(re.findall(r'\d+', privacy_response["text"])[0])
    
    # sentiment_response_score = int(json.loads(sentiment_response["text"])["Confidence"])
    sentiment_response_score = int(re.findall(r'\d+', sentiment_response["text"])[0])

    response = f'({accessibility_response_score}, {privacy_response_score}, {sentiment_response_score})'
    print(response)

    res = bayesian_network(privacy_response_score,sentiment_response_score,accessibility_response_score)

    return res

def user_output(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    accessibility_chain = get_output_accessibility_conversational_chain()
    privacy_chain = get_output_privacy_conversational_chain()
    sentiment_chain = get_output_sentiment_conversational_chain()

    output = llm3.invoke(user_question)
    print(output)

    accessibility_response = accessibility_chain.invoke(
        {"context":docs, "text": output}
        , return_only_outputs=True)
    privacy_response = privacy_chain.invoke(
        {"context":docs, "text": output}
        , return_only_outputs=True)
    sentiment_response = sentiment_chain.invoke(
        {"context":docs, "text": output}
        , return_only_outputs=True)

    # accessibility_response_score = int(json.loads(accessibility_response["text"])["Confidence"])    
    accessibility_response_score = int(re.findall(r'\d+', accessibility_response["text"])[0])
        
    # privacy_response_score = int(json.loads(privacy_response["text"])["Confidence"])
    privacy_response_score = int(re.findall(r'\d+', privacy_response["text"])[0])
    
    # sentiment_response_score = int(json.loads(sentiment_response["text"])["Confidence"])
    sentiment_response_score = int(re.findall(r'\d+', sentiment_response["text"])[0])

    response = f'({accessibility_response_score}, {privacy_response_score}, {sentiment_response_score})'
    print(response)

    res = bayesian_network(privacy_response_score,sentiment_response_score,accessibility_response_score)
    
    return res

def bayesian_network(privacy_score,sentiment_score,accessibility_score,user_violation_score=0):
    privacy_metric =  ""
    sentiment_metric = ""
    risq_metric = ""
    user_past_metric =  ""
    accessibility_metric = ""
    compliance_metric = ""

    if privacy_score <= 20:
        privacy_metric = "L"
    elif privacy_score > 20 and privacy_score <=60:
        privacy_metric = "M"
    else:
        privacy_metric = "H"

    if sentiment_score <= 30:
        sentiment_metric = "L"
    elif sentiment_score > 30 and sentiment_score <=70:
        sentiment_metric = "M"
    else:
        sentiment_metric = "H"

    if privacy_metric == "L" and sentiment_metric == "L":
        risq_metric = "L"
    elif privacy_metric == "L" and sentiment_metric == "M":
        risq_metric = "M"
    elif privacy_metric == "L" and sentiment_metric == "H":
        risq_metric = "H"
    elif privacy_metric == "M" and sentiment_metric == "L":
        risq_metric = "M"
    elif privacy_metric == "M" and sentiment_metric == "M":
        risq_metric = "M"
    elif privacy_metric == "M" and sentiment_metric == "H":
        risq_metric = "H"
    elif privacy_metric == "H" and sentiment_metric == "L":
        risq_metric = "H"
    elif privacy_metric == "H" and sentiment_metric == "M":
        risq_metric = "H"
    elif privacy_metric == "H" and sentiment_metric == "H":
        risq_metric = "H"
    
    if accessibility_score <= 20:
        accessibility_metric = "L"
    elif accessibility_score > 20 and accessibility_score <=60:
        accessibility_metric = "M"
    else:
        accessibility_metric = "H"

    if user_violation_score <= 40:
        user_past_metric = "L"
    elif user_violation_score > 40 and user_violation_score <= 70:
        user_past_metric = "M"
    else:
        user_past_metric = "H"

    if accessibility_metric == "L" and user_past_metric == "L":
        compliance_metric = "L"
    elif accessibility_metric == "L" and user_past_metric == "M":
        compliance_metric = "L"
    elif accessibility_metric == "L" and user_past_metric == "H":
        compliance_metric = "M"
    elif accessibility_metric == "M" and user_past_metric == "L":
        compliance_metric = "M"
    elif accessibility_metric == "M" and user_past_metric == "M":
        compliance_metric = "H"
    elif accessibility_metric == "M" and user_past_metric == "H":
        compliance_metric = "H"
    elif accessibility_metric == "H" and user_past_metric == "L":
        compliance_metric = "H"
    elif accessibility_metric == "H" and user_past_metric == "M":
        compliance_metric = "H"
    elif accessibility_metric == "H" and user_past_metric == "H":
        compliance_metric = "H"

    l_count = int(risq_metric == "L") + int(compliance_metric == "L") + int(sentiment_metric == "L")
    m_count = int(risq_metric == "M") + int(compliance_metric == "M") + int(sentiment_metric == "M")
    h_count = int(risq_metric == "H") + int(compliance_metric == "H") + int(sentiment_metric == "H")

    alert_risk = ""

    if l_count == 3:
        alert_risk = "L"
    elif l_count == 2 and m_count == 1:
        alert_risk = "L"
    elif l_count == 2 and h_count == 1:
        alert_risk = "M"
    elif m_count == 2 and l_count == 1:
        alert_risk = "M"
    elif m_count == 3:
        alert_risk = "H"
    elif m_count == 2 and h_count == 1:
        alert_risk = "H"
    elif h_count >= 2:
        alert_risk = "H"
    elif l_count == 1 and m_count == 1 and h_count == 1:
        alert_risk = "H"

    return alert_risk

def main():
    st.set_page_config("CompliBot 1.0")
    st.header("CompliBot: A Compliance Regulator for GPT")

    user_question = st.text_input("Ask a Question")

    if user_question:
        v1 = user_input(user_question)
        v2 = user_output(user_question)
        
        alert = "-1"
        if v1 == "L" and v2 == "L":
            alert = "L"
        elif v1 == "L" and v2 == "M":
            alert = "M"
        elif v1 == "L" and v2 == "H":
            alert = "H"
        elif v1 == "M" and v2 == "L":
            alert = "M"
        elif v1 == "M" and v2 == "M":
            alert = "H"
        elif v1 == "M" and v2 == "H":
            alert = "H"
        elif v1 == "H" and v2 == "L":
            alert = "M"
        elif v1 == "H" and v2 == "M":
            alert = "H"
        elif v1 == "H" and v2 == "H":
            alert = "H"
        
        st.write("Reply: ", alert)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your Policy Documents in PDF format and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()