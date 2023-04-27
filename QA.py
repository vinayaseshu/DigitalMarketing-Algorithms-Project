from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
import os
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains import SequentialChain


@st.cache_resource
def setup():
    
    os.environ['OPENAI_API_KEY'] = st.secrets['open_ai']
    #Loadind data from documents
    loader = DirectoryLoader('./Documents', glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()
    
    #Splitting data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    
    #Setup OpenAI and Load vectors to ChromaDB
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embeddings)
    llm = OpenAI(temperature=0.3, openai_api_key=st.secrets['open_ai'])

    return vectordb,llm 

@st.cache_resource
def prompt_setup():
    #################################################
    prompt_template1 = """ Prompt: Use the following pieces of context to answer the question

    {context}

    Question: {question}

    Answer:"""
    first_prompt = PromptTemplate(
        template=prompt_template1, input_variables=["context", "question"]
    )

    ###################################################
    prompt_template2 = """Here is a statement:
            {statement}
            Make the statement sound genuine also tell them to leave us a review if this response helped them .\n\n"""
    second_prompt = PromptTemplate(input_variables=["statement"], template=prompt_template2)
    ###################################################

    question_chain = LLMChain(llm=llm, prompt=first_prompt , output_key = 'statement')
    assumptions_chain = LLMChain(llm=llm, prompt=second_prompt, output_key = 'response')

    overall_chain = SequentialChain(
        chains=[question_chain, assumptions_chain],
        input_variables=['context', "question"],
        # Here we return multiple variables
        output_variables=["statement", "response"],
        verbose=False)
    
    return overall_chain


def answer_query(query,overall_chain):
    docsearch = vectordb.similarity_search(query, k=8)

    response = overall_chain({'context':docsearch, 'question':query})['response']
    
    return response

vectordb, llm = setup()
overall_chain = prompt_setup()

st.title('Knowledge base for questions & answers')

st.header('Looking for specific info?')

st.markdown('Are you tired of sifting through countless reviews trying to find the information you need about a product? Look no further than our new Chat with Reviewers feature! Get direct access to product reviewers and customers who have used the product you\'re interested in. Ask them questions, get their honest opinions, and make an informed decision about your purchase. Say goodbye to uncertainty and hello to confidence in your buying decisions with Chat with Reviewers.')

query = st.text_input(label='Question ?', key='query',value='Welcome customers')

ask = st.button(label='Ask', key='run')

if ask: 
    response = answer_query(query,overall_chain)

    st.markdown(response)
