from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs,download_hugging_face_embeddings,text_split




load_dotenv()
# Set API keys from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Load and process documents
extracted_data = load_pdf_files(data="data/")
filter_data = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(filter_data)

# Prepare embeddings
embeddings = download_hugging_face_embeddings()


# Initialize Pinecone client
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


# Define index name
index_name = "medical-chatbot"

# Check if index exists
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # must match your embedding model output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="eu-north-1"),
    )

# Get reference to the index
index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name,
)
