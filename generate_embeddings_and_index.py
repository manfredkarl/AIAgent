from azure.search.documents import SearchIndexingBufferedSender
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import os

load_dotenv(override=True) # take environment variables from .env.

az_openai_client = AzureOpenAI(  
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)  

ai_search_endpoint = os.environ["AI_SEARCH_SERVICE_ENDPOINT"]
ai_search_credential = AzureKeyCredential(os.environ["AI_SEARCH_SERVICE_QUERY_KEY"])
ai_search_index_name = os.environ["INDEX_NAME"]

# Lee el archivo JSON
filename = 'data/description_products.json'

with open(filename, 'r', encoding="utf-8") as f:
    documents = json.load(f)

print(f"Total of documents to upload: {len(documents)}")
lote=[]




# Crea un cliente de índice
with SearchIndexingBufferedSender(
             endpoint=ai_search_endpoint,
             index_name=ai_search_index_name,
             credential=ai_search_credential
        ) as batch_client:
    
    for i, doc in enumerate(documents):
        doc["id"] = str(i)
        print(f'[{i}]: question: {doc["question"]}, answer: {doc["answer"]}')

        doc["question_embeddings"] = az_openai_client.embeddings.create(input = [doc["question"]], model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME_ADA")).data[0].embedding
        doc["answer_embeddings"] = az_openai_client.embeddings.create(input = [doc["answer"]], model=os.getenv("AZURE_OPENAI_EMBEDDING_NAME_ADA")).data[0].embedding
        lote.append(doc)

    try:
        print(f'Indexing until document {i}...')
        batch_client.upload_documents(documents=lote)

    except Exception as ex:
        print(ex)
