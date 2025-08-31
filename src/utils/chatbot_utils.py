import os 
import sys
from typing import Any

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.utils.logger import logging
from src.utils.exception import Custom_exception
from dotenv import load_dotenv
load_dotenv()



class BuildRetrievalchain:
    """
    contains helper function for creating chatbot
    embeddings, llm, prompt, vector_store, retriever, retrieval_chain
    """

    def __init__(self):
        pass

    def load_embeddings(self):
        try:
            logging.info("Initializing NVIDIA Embeddings.")
            embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2",
                                        api_key=os.getenv("NVIDIA_API_KEY"),
                                        truncate="NONE")
                
            logging.info("Embeddings initialized successfully.")
            return embeddings
            
        except Exception as e:
            logging.error(f"Error initializing embeddings: {str(e)}")
            raise Custom_exception(e, sys)



    def load_llm(self):
        try:
            logging.info("Initializing Llama2 model with Groq")
            llm = ChatGroq(temperature=0.6,
                        model_name="llama-3.3-70b-versatile",
                        groq_api_key=os.getenv("GROQ_API_KEY"),
                        max_tokens=4096)
            
            logging.info("LLM initialized successfully")
            return llm
            
        except Exception as e:
            logging.error(f"Error initializing LLM: {str(e)}")
            raise Custom_exception(e, sys)
        


    def setup_prompt(self):
        try:
            logging.info("Creating prompt template")
            system_prompt = """
You are a knowledgeable and friendly personal assistant for an e-commerce store.

When recommending products, make your answers visually appealing and easy to read:
- Use bullet points or numbered lists for product options.
- Add a blank line between each product.
- Show only the most important details: Brand, Product Name, Price, MRP, Offer.
- Keep each product's info concise and on separate lines.
- If there are multiple products, limit to 3-4 at a time and say "Let me know if you'd like to see more options!"
- Start with a short, friendly intro and end with a helpful closing line.

Example format for recommendations:

Here are some sarees you might like:


1. Brand: Sugathari
    Product: Women's Banarasi Saree Pure Kanjivaram Silk Saree
    Price: £5.22 (MRP: £21.84, 76% off)

2. Brand: Sugathari
    Product: Women's Banarasi Saree Pure Kanjivaram Silk Saree (Cotton)
    Price: £4.74 (MRP: £21.84, 78% off)

Let me know if you want more details or to place an order!

---

Other instructions:
- ONLY provide information that is explicitly mentioned in the context provided.
 - Always display all prices in GBP (£), never in dollars or any other currency.
- If specific details (prices, brands, materials) of a product are not in the context, DO NOT make them up.
- If you're unsure or don't have enough information, say so directly.
- When asked to recommend a product under a certain price, only show products that meet the user's condition.
- Format prices exactly as they appear in the context, don't modify them.
- For invoices, keep the format clean and easy to read, using lines and spacing for clarity.
- Be professional, brief, and visually clear in your responses.

Current context about our products and inventory:
{context}
"""
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            logging.info("Prompt template has been created")
            return prompt
        except Exception as e:
            logging.error(f"Error creating prompt: {str(e)}")
            raise Custom_exception(e, sys)


    def load_vectorstore(self, embeddings):
        try:
            logging.info("Loading vectorstore ")
            vector_store = PineconeVectorStore.from_existing_index(index_name="ecommerce-chatbot-project",
                                                                   embedding=embeddings)

            logging.info("Successfully loaded vectorstore")
            return vector_store
        
        except Exception as e:
            raise Custom_exception(e, sys)
        

    def build_retriever(self, vector_store: PineconeVectorStore):
        try:
            logging.info("Initializing vector_store as retriever")
            retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                                  search_kwargs={"k": 5, "score_threshold": 0.5})
            logging.info("Retriever has been initialized")
            return retriever
        except Exception as e:
            logging.info(f"Error initializing retriever: {str(e)}")
            raise Custom_exception(e, sys)
        


    def build_chains(self, llm: Any, prompt: ChatPromptTemplate, retriever: Any):
        try:
            logging.info("Creating stuff document chain...")
            doc_chain = create_stuff_documents_chain(llm=llm, 
                                                    prompt=prompt,
                                                    output_parser=StrOutputParser(),
                                                    document_variable_name="context")
            
            logging.info("Creating retrieval chain...")
            retrieval_chain = create_retrieval_chain(retriever=retriever, 
                                                    combine_docs_chain=doc_chain)
            
            logging.info("Chains created successfully")
            return retrieval_chain
        
        except Exception as e:
            logging.info(f"Error creating chains {str(e)}")
            raise Custom_exception(e, sys)
        


    def build_retrieval_chain(self):
        try:
            embeddings = self.load_embeddings()
            llm = self.load_llm()
            prompt = self.setup_prompt()
            vector_store = self.load_vectorstore(embeddings)
            retriever = self.build_retriever(vector_store)
            retrieval_chain = self.build_chains(llm, prompt, retriever)

            return retrieval_chain
        except Exception as e:
            raise Custom_exception(e, sys)
        
    
    

class BuildChatbot:
    def __init__(self):
        self.store = {}  # Persistent dictionary to maintain chat history


    def get_session_id(self, session_id: str) -> BaseChatMessageHistory:
        """creates and retrieves a chat history session."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]


    def initialize_chatbot(self):
        """Initializes the chatbot with session memory."""
        utils = BuildRetrievalchain()
        retrieval_chain = utils.build_retrieval_chain()

        chatbot = RunnableWithMessageHistory(runnable=retrieval_chain,
                                             get_session_history=self.get_session_id,
                                             input_messages_key="input",
                                             history_messages_key="chat_history",
                                             output_messages_key="answer")

        return chatbot
