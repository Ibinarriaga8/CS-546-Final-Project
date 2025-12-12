# rag.py


import abc
import getpass
import os
import bs4
from typing import List, Any
from dataclasses import dataclass, field


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, InMemoryVectorStore
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.documents import Document
from langchain_classic import hub

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chat_models import init_chat_model # This is a helper, we'll use ChatGroq directly

from typing import Union


if not os.environ.get("GROQ_API_KEY"):
    print("GROQ_API_KEY not found in environment variables.")
    os.environ["GROQ_API_KEY"] = "" # Enter your Groq API key here or set it as an environment variable


@dataclass
class RAGConfig:
    """
    Configuration class for the RAG system to make hyperparameters configurable.
    """
    # Loader Config
    
    urls: List[str] = field(default_factory=lambda: ["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    texts: List[str] = field(default_factory=list)
    # Splitter Config
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retriever Config
    top_k: int = 4
    
    # Generator Config
    llm_model_name: str = "groq/compound" # "llama-3.1-8b-instant" (or any Groq model)
    llm_provider: str = "groq"
    llm_temperature: float = 0.1

    # Prompt Config
    prompt_hub_path: str = "rlm/rag-prompt"
    custom_prompt_template: ChatPromptTemplate | None = None

    # Embeddings Config
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # Vector Store Config
    persist_directory: str = "./chroma_db"


# Abstract Base Classes (Interfaces) for RAG Components

class BaseLoader(abc.ABC):
    """
    Abstract class for loading documents.
    """
    @abc.abstractmethod
    def load(self) -> List[Document]:
        """
        Loads documents from a source.
        """
        pass

class BaseIndexer(abc.ABC):
    """
    Abstract class for indexing documents.
    """
    @abc.abstractmethod
    def index_documents(self, docs: List[Document]) -> Any:
        """
        Splits, embeds, and stores documents in a vector store.
        Returns the vector store object.
        """
        pass

class BaseRetriever(abc.ABC):
    """
    Abstract class for retrieving documents.
    """
    @abc.abstractmethod
    def get_retriever(self, vector_store: Any) -> Runnable:
        """
        Returns a retriever runnable (e.g., from a vector store).
        """
        pass

class BaseGenerator(abc.ABC):
    """
    Abstract class for generating responses.
    """
    @abc.abstractmethod
    def get_chain(self, retriever: Runnable) -> Runnable:
        """
        Returns a runnable RAG chain.
        """
        pass

# Concrete Implementations 

# Loaders 

class ConfigurableLoader(BaseLoader):
    """
    Concrete implementation for loading web pages using user's bs4 config.
    """
    def __init__(self, urls: List[str], texts: List[str] = []):
        self.urls = urls
        self.texts = texts
        self.loader = WebBaseLoader(
            web_paths=self.urls,
        )


    def load(self) -> List[Document]:
        docs = []

        if self.urls:
            print(f"Loading documents from URLs: {self.urls}")

            try:
                url_loader = WebBaseLoader(web_paths=self.urls)
                url_docs = url_loader.load()
                docs.extend(url_docs)
                print(f"Loaded {len(url_docs)} URL documents.")
            except Exception as e:
                print(f"URL loading failed: {e}")

        if self.texts:
            print(f"Loading {len(self.texts)} in-memory text documents...")
            text_docs = [
                Document(page_content=t, metadata={"source": f"text_{i}"})
                for i, t in enumerate(self.texts)
            ]
            docs.extend(text_docs)

        return docs

class ChromaIndexer(BaseIndexer):
    """
    Concrete implementation for indexing using ChromaDB (Persistent).
    """
    def __init__(self, embeddings_model, text_splitter, persist_directory: str):
        self.embeddings = embeddings_model
        self.text_splitter = text_splitter
        self.persist_directory = persist_directory
        self.vector_store = None

    def index_documents(self, docs: List[Document]) -> Chroma:
        print(f"Splitting {len(docs)} documents...")
        split_docs = self.text_splitter.split_documents(docs)
        
        print(f"Creating persistent vector store at {self.persist_directory}...")
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Indexing complete.")
        return self.vector_store

class InMemoryIndexer(BaseIndexer):
    """
    Concrete implementation for indexing using InMemoryVectorStore.
    """
    def __init__(self, embeddings_model, text_splitter):
        self.embeddings = embeddings_model
        self.text_splitter = text_splitter
    
    def index_documents(self, docs: List[Document]) -> InMemoryVectorStore:
        print(f"Splitting {len(docs)} documents...")
        split_docs = self.text_splitter.split_documents(docs)
        
        print(f"Creating in-memory vector store...")
        vector_store = InMemoryVectorStore(embedding=self.embeddings)
        vector_store.add_documents(documents=split_docs)
        
        print(f"Indexing complete. {len(split_docs)} chunks added to memory.")
        return vector_store

# Retrievers 

class ChromaRetriever(BaseRetriever):
    """
    Concrete implementation for retrieving from an existing ChromaDB.
    """
    def __init__(self, embeddings_model, persist_directory: str, top_k: int):
        self.persist_directory = persist_directory
        self.embeddings = embeddings_model
        self.top_k = top_k

    def get_retriever(self, vector_store: Any = None) -> Runnable:
        """
        Loads the persistent vector store from disk.
        Ignores the vector_store argument.
        """
        print(f"Loading persistent vector store from {self.persist_directory}...")
        try:
            vector_store_from_disk = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            return vector_store_from_disk.as_retriever(
                search_kwargs={"k": self.top_k}
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise ValueError("Could not load persistent vector store. Run indexing first.")

class InMemoryRetriever(BaseRetriever):
    """
    Concrete implementation for retrieving from an InMemoryVectorStore.
    """
    def __init__(self, top_k: int):
        self.top_k = top_k

    def get_retriever(self, vector_store: Any) -> Runnable:
        """
        Uses the provided in-memory vector store.
        """
        if not isinstance(vector_store, InMemoryVectorStore):
            raise ValueError(
                "InMemoryRetriever requires an InMemoryVectorStore object. "
                "Please run the InMemoryIndexer first."
            )
        print("Using in-memory vector store for retrieval...")
        return vector_store.as_retriever(search_kwargs={"k": self.top_k})


# Generators 

class LangChainGenerator(BaseGenerator):
    """
    Concrete implementation for generating responses using a LangChain model.
    """
    def __init__(self, llm, prompt_template):
        self.llm = llm
        self.prompt = prompt_template

    def get_chain(self, retriever: Runnable) -> Runnable:
        """
        Builds the LCEL RAG chain.
        """
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

# Main RAG System 

class RAG:
    """
    Main RAG system class that composes the Indexer, Retriever, and Generator.
    This class is now flexible and works with both persistent and in-memory stores.
    """
    def __init__(
        self,
        loader: BaseLoader,
        indexer: BaseIndexer,
        retriever: BaseRetriever,
        generator: BaseGenerator
    ):
        self.loader = loader
        self.indexer = indexer
        self.retriever = retriever
        self.generator = generator
        self.vector_store = None
        self.rag_chain = None

    def setup_pipeline(self):
        """
        Public method to run the loading, indexing, and chain-loading pipeline.
        """
        # 1. Load Documents
        docs = self.loader.load()
        
        # 2. Index Documents
        print("--- Indexing Documents ---")
        self.vector_store = self.indexer.index_documents(docs)
        print("--- Indexing Complete ---")

        # 3. Load RAG Chain
        print("--- Loading RAG Chain ---")
        retriever_runnable = self.retriever.get_retriever(self.vector_store)
        self.rag_chain = self.generator.get_chain(retriever_runnable)
        print("--- RAG System is Ready ---")

    def ask(self, query: str):
        """
        Public method to ask a question to the RAG system.
        """
        if not self.rag_chain:
            raise ValueError("RAG chain not loaded. Call .setup_pipeline() first.")
        
        print(f"\nQuery: {query}")
        response = self.rag_chain.invoke(query)
        
        print(f"Answer: {response}")
        return response

# Example Usage
def main_inmemory_groq():
    """
    Main function to demonstrate the RAG system using the user's
    Groq, HuggingFace, and InMemoryVectorStore configuration.
    """

    # 1. Create a configuration object
    config = RAGConfig(
        # --- UPDATED URLS (ENGLISH, ML TOPIC) ---
        urls=[
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://developers.google.com/machine-learning/crash-course/ml-intro",
            "https://www.ibm.com/topics/machine-learning"
        ],
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        llm_model_name="llama-3.1-8b-instant",
        llm_provider="groq",
        llm_temperature=0.9,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 2. Instantiate all components based on the config

    # Loader
    loader = ConfigurableLoader(urls=config.urls)

    # Embeddings
    print(f"Loading embedding model: {config.embedding_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )

    # LLM
    print(f"Loading LLM: {config.llm_model_name} from {config.llm_provider}")
    llm = ChatGroq(
        model=config.llm_model_name,
        temperature=config.llm_temperature
    )

    # Prompt
    print(f"Pulling prompt from Hub: {config.prompt_hub_path}")
    prompt = hub.pull(config.prompt_hub_path)

    # Indexer (In-Memory)
    indexer = InMemoryIndexer(
        embeddings_model=embeddings,
        text_splitter=text_splitter
    )

    # Retriever (In-Memory)
    retriever = InMemoryRetriever(top_k=config.top_k)

    # Generator
    generator = LangChainGenerator(llm=llm, prompt_template=prompt)

    # 3. Instantiate the main RAG system
    rag_system = RAG(
        loader=loader,
        indexer=indexer,
        retriever=retriever,
        generator=generator
    )

    # 4. Run the setup pipeline
    rag_system.setup_pipeline()

    # 5. Ask questions (ML TOPIC, ENGLISH)

    rag_system.ask("What is machine learning and how does it differ from traditional programming?")
    rag_system.ask("What are the main types of machine learning?")
    rag_system.ask("What is supervised learning and when is it typically used?")
    rag_system.ask("How does unsupervised learning differ from supervised learning?")
    rag_system.ask("What is the role of a loss function in machine learning?")
    rag_system.ask("What is overfitting and how can it be prevented?")
    rag_system.ask("What is the difference between training data and test data?")
    rag_system.ask("What is a feature in the context of machine learning?")
    rag_system.ask("How does gradient descent work at a high level?")
    rag_system.ask("What are some common real-world applications of machine learning?")

def main_chroma_ollama():
    """
    Main function to demonstrate the OOPS RAG system using
    Ollama and persistent ChromaDB storage.
    """
    
    # 1. Define Core Components
    print("Using Ollama models...")
    llm = ChatOllama(model="llama3")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    config = RAGConfig(
        urls=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
        chunk_size=3000,
        chunk_overlap=200,
        top_k=3,
        llm_model_name="llama3",
        llm_provider="ollama",
        llm_temperature=0.7
    )

    # 2. Instantiate components
    loader = ConfigurableLoader(urls=config.urls)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, 
        chunk_overlap=config.chunk_overlap
    )
    
    prompt = hub.pull(config.prompt_hub_path) # Use the same hub prompt
    
    # Indexer (Chroma)
    indexer = ChromaIndexer(
        embeddings_model=embeddings,
        text_splitter=text_splitter,
        persist_directory=config.persist_directory
    )
    
    # Retriever (Chroma)
    retriever = ChromaRetriever(
        embeddings_model=embeddings,
        persist_directory=config.persist_directory,
        top_k=config.top_k
    )
    
    # Generator
    generator = LangChainGenerator(llm=llm, prompt_template=prompt)

    # 3. Instantiate the main RAG system
    rag_system = RAG(
        loader=loader,
        indexer=indexer,
        retriever=retriever,
        generator=generator
    )
    
    # 4. Run the setup pipeline
    rag_system.setup_pipeline()
    
    # 5. Ask questions
    rag_system.ask("What are the main components of an LLM agent?")
    rag_system.ask("What is task decomposition?")




if __name__ == "__main__":
    main_inmemory_groq()
    # main_chroma_ollama()# rl_rag.py