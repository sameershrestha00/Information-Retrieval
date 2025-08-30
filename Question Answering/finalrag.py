import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
assert GOOGLE_API_KEY, "GOOGLE_API_KEY not found in .env file"

# Initialize LLM and memory
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Load PDF and create retriever
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever()

# Initialize QA chain
qa_chain = None
def initialize_rag(file_path):
    global qa_chain
    retriever = load_pdf(file_path)
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Ask questions
def pdf_qa_tool(question: str) -> str:
    if qa_chain is None:
        return "Please initialize RAG with a PDF first."
    return qa_chain.run(question)

#main 
if __name__ == "__main__":
    
    pdf_path = "temp.pdf"  
    initialize_rag(pdf_path)
    print(f"Loaded PDF: {pdf_path}")
    print("RAG system ready! Ask questions about your PDF (type 'exit' to quit).")

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        answer = pdf_qa_tool(question)
        print("Answer:", answer)
