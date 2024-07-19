import os
import json
import boto3
from requests_aws4auth import AWS4Auth
from PyPDF2 import PdfFileReader
from langchain.vectorstores import OpenSearchVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.retrievers import VectorStoreRetriever

# AWS configurations
region = 'your-region'  # e.g., 'us-west-2'
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

host = 'https://your-opensearch-domain.us-west-2.es.amazonaws.com'  # The OpenSearch domain endpoint
index = 'your-index'

# Initialize the OpenSearch vector store
vector_store = OpenSearchVectorStore(
    host=host,
    index=index,
    aws_auth=awsauth
)

print("Vector store created.")

# Function to read PDF files and extract text
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()
    return text

# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings()

# Directory containing the PDF files
data_dir = './data_files/'

# Process each PDF file in the directory
for filename in os.listdir(data_dir):
    if filename.endswith('.pdf'):
        file_path = os.path.join(data_dir, filename)
        text = read_pdf(file_path)
        
        # Create a document object
        document = Document(content=text)
        
        # Generate embeddings for the document text
        embedding = embeddings_model.embed(document.content)
        
        # Add the document and its embedding to the vector store
        vector_store.add_document(document=document, embedding=embedding)

print("Documents added to vector store.")

# Initialize the retriever
retriever = VectorStoreRetriever(vector_store=vector_store)

# Define your query text
query_text = "example query text"

# Generate embeddings for the query text
query_embedding = embeddings_model.embed(query_text)

# Retrieve the nearest neighbors
results = retriever.retrieve(query_embedding, k=10)

# Print the results
for result in results:
    print(f"Document ID: {result['id']}, Score: {result['score']}, Content: {result['document'].content}")
