import os
import csv
import logging
import requests
from huggingface_hub import InferenceClient
import streamlit as st
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForMaskedLM ,AutoModel
from transformers import AutoModelForCausalLM
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain, RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings , HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# OpenAI API key
openai_api_key = 'sk-proj-9rAqRYY5iSqw4ad9kut7T3BlbkFJ0MBPmcsamxNPIZ7sthgP'
llm = ChatOpenAI(api_key=openai_api_key)

# Hugging Face API key
os.environ["HUGGINGFACE_API_KEY"] = "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"

from huggingface_hub import InferenceClient

client = InferenceClient(
    "mistralai/Mistral-7B-Instruct-v0.1",
    token="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",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=True,
):
    print(message.choices[0].delta.content, end="")



# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# Function to load LLM
def load_llm():
    model = AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    return CTransformers(
        model="Mistral-7B-Instruct-v0.1",
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0.3
    )

# Function to process the PDF file
def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''.join(page.page_content for page in data)
    
    splitter_ques_gen = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen

# Function to generate text using a specified model
def generate_text(prompt):
    try:
        model = pipeline("text-generation", model="gpt-2")
        response = model(prompt, max_length=50)
        return response[0]['generated_text']
    except Exception as e:
        print(f"Error during text generation: {e}")
        raise

# Function to get vector store
def get_vectorstore(text_chunks):
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
    print(f"Using API Key: {api_key}")  # Debugging line

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", api_key=api_key)
    return embeddings

# Function to generate questions and answers
def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)
    llm = load_llm()

    prompt_template = """
    You are an expert at creating questions based on coding materials and documentation.
    Your goal is to prepare a coder or programmer for their exam and coding tests.
    You do this by asking questions about the text below:
    ------------
    {text}
    ------------
    Create questions that will prepare the coders or programmers for their tests.
    Make sure not to lose any important information.
    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    You are an expert at creating practice questions based on coding material and documentation.
    Your goal is to help a coder or programmer prepare for a coding test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------
    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(input_variables=["existing_answer", "text"], template=refine_template)

    ques_gen_chain = LLMChain(llm=llm, prompt=PROMPT_QUESTIONS)
    refined_ques_gen_chain = LLMChain(llm=llm, prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run({"text": "\n".join(doc.page_content for doc in document_ques_gen)})

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = llm
    ques_list = [q for q in ques.split("\n") if q.endswith('?') or q.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever())

    return answer_generation_chain, ques_list

# Function to generate CSV file
def get_csv(file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    os.makedirs(base_folder, exist_ok=True)
    output_file = os.path.join(base_folder, "QA.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])

        for question in ques_list:
            answer = answer_generation_chain.run(question)
            csv_writer.writerow([question, answer])

    return output_file

# Streamlit app
st.title("PDF Q&A Generator")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file is not None:
    pdf_filename = os.path.join("static/docs", uploaded_file.name)
    with open(pdf_filename, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"File {uploaded_file.name} uploaded successfully!")

    if st.button("Generate Q&A"):
        with st.spinner("Processing..."):
            output_file = get_csv(pdf_filename)
            st.success("Q&A generation completed!")
            st.download_button(label="Download Q&A CSV", data=open(output_file, "rb"), file_name="QA.csv", mime="text/csv")
