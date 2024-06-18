import streamlit as st

# Opensource Embeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

# Vector Store for Vector Embeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Load RetrievalQA from langchain as it provides a simple interface to interact with the LLM.
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

# Imports for Data Ingestion
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# call llama LLM locally
from langchain.chat_models import ChatOpenAI
import time

OAI_KEY = os.environ["OAI_KEY"]

# Load the PDFs from the directory
def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    docs = text_splitter.split_documents(documents)
    return docs

# Vector Store for Vector Embeddings
def setup_vector_store(documents):
    # Create a vector store using FAISS from the documents and the embeddings
    vector_store = FAISS.from_documents(
        documents,
        embedding_function,
    )
    # Save the vector store locally
    vector_store.save_local("faiss_index")

# Create a prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the answer is not within the context knowledge, kindly state that you do not know, rather than attempting to fabricate a response.
2. If you find the answer, please craft a detailed and concise response to the question at the end. Aim for a summary of max 500 words, ensuring that your explanation is thorough.

{context}

Question: {question}
Helpful Answer:"""

# Now we use langchain PromptTemplate to create the prompt template for our LLM
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create FAISS from the documents
setup_vector_store(data_ingestion())

#initialize the LLM object
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OAI_KEY,
)

# create the open-source embedding function
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large",api_key=OAI_KEY)

faiss_index = FAISS.load_local(
    "faiss_index", embeddings=embedding_function, allow_dangerous_deserialization=True
)
retriever_FASS = faiss_index.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)


# Create a RetrievalQA chain and invoke the LLM
def get_response(llm, db, query):
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # retriever=retriever,
        retriever=retriever_FASS,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    response = retrieval_qa.invoke(query)
    return response["result"]


# Streamed response emulator
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("PodGenie :headphones:")
st.info("A GenAI conversation chat buddy to discuss your favourite podcasts _SuperDataScience_", icon="🤖")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # llm = load_llm()
        response = st.write_stream(
            response_generator(get_response(llm, faiss_index, prompt))
        )
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
