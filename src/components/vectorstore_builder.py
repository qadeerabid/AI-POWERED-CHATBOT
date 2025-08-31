import os 
import sys 
import time
from typing import List
from dataclasses import dataclass

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from src.utils.logger import logging
from src.utils.exception import Custom_exception
from dotenv import load_dotenv

load_dotenv()


@dataclass
class VectorStoreBuilderConfig:
    is_airflow = os.getenv("IS_AIRFLOW", "false").lower() == "true"

    if is_airflow:
        path = "/opt/airflow/artifacts/data_cleaned.csv"

    else:
        path = "artifacts/data_cleaned.csv"

class VectorStoreBuilder:
    """
    Load data 
    Create embeddings 
    Create vector store and return the vector store 
    """

    def __init__(self):
        self.vectorstore_builder_config = VectorStoreBuilderConfig()
        self.nvidia_api_key = os.getenv("NVIDIA_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        print(f"[DEBUG] NVIDIA_API_KEY: {self.nvidia_api_key}")
        print(f"[DEBUG] PINECONE_API_KEY: {self.pinecone_api_key}")
        if not self.nvidia_api_key or not self.pinecone_api_key:
            raise ValueError("Required API keys not set")





    def load_data(self, data_paths: list) -> List[Document]:
        """
        Load and combine documents from multiple CSV files. Truncate text to 512 characters for embedding.
        Enhanced logging: checks file existence, logs document counts per file, and errors.
        """
        all_docs = []
        for data_path in data_paths:
            try:
                if not os.path.exists(data_path):
                    logging.error(f"File not found: {data_path}")
                    print(f"[ERROR] File not found: {data_path}")
                    continue
                logging.info(f"Loading data from {data_path}")
                print(f"[INFO] Loading data from {data_path}")
                loader = CSVLoader(file_path=data_path,
                                   encoding="utf-8",
                                    csv_args={"delimiter": ",",
                                              "quotechar": '"'})
                docs = loader.load()
                # Truncate document text to 512 characters
                for doc in docs:
                    if hasattr(doc, 'page_content'):
                        doc.page_content = doc.page_content[:512]
                logging.info(f"Sample data from {data_path}: {docs[:2]}")
                print(f"[INFO] Loaded {len(docs)} documents from {data_path}")
                logging.info(f"Loaded {len(docs)} documents from {data_path}.")
                all_docs.extend(docs)
            except Exception as e:
                logging.error(f"Error in loading data from {data_path}: {str(e)}")
                print(f"[ERROR] Error in loading data from {data_path}: {str(e)}")
                continue
        print(f"[INFO] Total combined documents: {len(all_docs)}")
        logging.info(f"Total combined documents: {len(all_docs)}")
        return all_docs
    


    def create_embeddings(self) -> NVIDIAEmbeddings:
        try:
            logging.info("Initializing NVIDIA Embeddings.")
            embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embedqa-mistral-7b-v2",
                api_key=self.nvidia_api_key,
                truncate="NONE")
            
            logging.info("Embeddings initialized successfully.")
            return embeddings
        
        except Exception as e:
            logging.error(f"Error initializing embeddings: {str(e)}")
            raise Custom_exception(e, sys)
    



    def create_vector_store(self, documents: List[Document], 
                            embeddings: NVIDIAEmbeddings, 
                            index_name: str = 'ecommerce-chatbot-project') -> PineconeVectorStore:
        try:
            print(f"[DEBUG] Attempting to create Pinecone index: {index_name} with dimension 4096")
            logging.info(f"Connecting to Pinecone and creating index: {index_name}")
            pc = Pinecone(api_key=self.pinecone_api_key)

            # Check if index already exists
            existing_indexes = [idx['name'] for idx in pc.list_indexes()]
            if index_name not in existing_indexes:
                try:
                    pc.create_index(name=index_name,
                                    dimension=4096,
                                    metric="cosine",
                                    spec=ServerlessSpec(cloud="aws", region="us-east-1"))
                    print(f"[DEBUG] Index creation requested.")
                    time.sleep(10)
                except Exception as e:
                    print(f"[DEBUG] Exception during index creation: {e}")
                    raise
            else:
                print(f"[DEBUG] Index '{index_name}' already exists. Skipping creation.")

            index = pc.Index(index_name)
            time.sleep(10)

            # Proceed directly to uploading new data (no manual delete step)

            initial_stats = index.describe_index_stats()
            print(f"[DEBUG] Index status before uploading: {initial_stats}")
            logging.info(f"Index status before uploading: {initial_stats}")

            vector_store = PineconeVectorStore.from_documents(documents=documents,
                                                              index_name=index_name, 
                                                              embedding = embeddings)

            final_stats = index.describe_index_stats()
            logging.info(f"Index status after uploading: {final_stats}")

            logging.info(f"Successfully created vector store with {len(documents)} documents")
            return vector_store
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            raise Custom_exception(e, sys)
        



    def run_pipeline(self) -> PineconeVectorStore:
        try:
            logging.info("Starting vectorstore pipeline (product data only)")
            # Only use product data
            data_paths = [
                self.vectorstore_builder_config.path
            ]
            docs = self.load_data(data_paths)
            embeddings = self.create_embeddings()
            vector_store = self.create_vector_store(docs, embeddings)

            logging.info("Vectorstore pipeline completed successfully (product data only)")
            return vector_store
        except Exception as e:
            logging.error(f"Error in pipeline execution: {str(e)}")
            raise Custom_exception(e, sys)





if __name__ == "__main__":
    builder = VectorStoreBuilder()
    builder.run_pipeline()