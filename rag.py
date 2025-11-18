import abc
import getpass
import os
import bs4
from typing import List, Any
from dataclasses import dataclass, field

# --- Core LangChain Imports ---

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

# --- User-Specified Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chat_models import init_chat_model # This is a helper, we'll use ChatGroq directly

# --- API Key Setup (from user snippet) ---
if not os.environ.get("GROQ_API_KEY"):
    print("GROQ_API_KEY not found in environment variables.")
    os.environ["GROQ_API_KEY"] =  input("Enter API key for Groq: ")


# --- Configuration ---

@dataclass
class RAGConfig:
    """
    Configuration class for the RAG system to make hyperparameters configurable.
    """
    # Loader Config
    urls: List[str] = field(default_factory=lambda: ["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    
    # Splitter Config
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retriever Config
    top_k: int = 4
    
    # Generator Config
    llm_model_name: str = "llama-3.1-8b-instant"
    llm_provider: str = "groq"
    llm_temperature: float = 0.7
    prompt_hub_path: str = "rlm/rag-prompt"

    # Embeddings Config
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"

    # Vector Store Config
    persist_directory: str = "./chroma_db"


# --- Abstract Base Classes (Interfaces) ---

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

# --- Concrete Implementations ---

# --- Loaders ---

class ConfigurableLoader(BaseLoader):
    """
    Concrete implementation for loading web pages using user's bs4 config.
    """
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.loader = WebBaseLoader(
            web_paths=self.urls,
        )

    def load(self) -> List[Document]:
        print(f"Loading documents from: {self.urls}")
        docs = self.loader.load()
        print(f"Loaded {len(docs)} documents.")
        return docs

# --- Indexers ---

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

# --- Retrievers ---

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


# --- Generators ---

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

# --- Main RAG System Class (Composition) ---

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

# --- Example Usage ---

def main_inmemory_groq():
    """
    Main function to demonstrate the RAG system using the user's
    Groq, HuggingFace, and InMemoryVectorStore configuration.
    
    *** MODIFIED TO BE ABOUT "CÓDIGO PENAL ESPAÑOL" ***
    """
    
    # 1. Create a configuration object
    config = RAGConfig(
        # --- MODIFIED URLS ---
        urls=[
            "https://www.boe.es/buscar/act.php?id=BOE-A-1995-25444", # BOE - Texto Consolidado
            "https://es.wikipedia.org/wiki/C%C3%B3digo_Penal_de_Espa%C3%B1a", # Wikipedia
            "https://www.conceptosjuridicos.com/codigo-penal/" # Legal concepts summary
        ],
        # --- END MODIFICATION ---
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
    
    # 4. Run the setup pipeline (load, index, create chain)
    rag_system.setup_pipeline()
    
    # 5. Ask questions
    # --- MODIFIED QUESTIONS ---
# --- Preguntas sobre la Exposición de Motivos (Preamble) ---
    rag_system.ask("Según la exposición de motivos, ¿por qué se considera el Código Penal como una 'Constitución negativa'?")
    rag_system.ask("¿Cuál es el objetivo principal de la reforma del sistema de penas mencionada en la exposición de motivos?")
    rag_system.ask("¿Cómo cambia la nueva regulación de los delitos contra la libertad sexual el bien jurídico protegido?")

    # --- Preguntas sobre el Título Preliminar (Artículos 1-9) ---
    rag_system.ask("Según el Artículo 1, ¿puede castigarse una acción que no esté prevista como delito por ley anterior a su perpetración?")
    rag_system.ask("Basado en el Artículo 5, ¿es posible que exista una pena si no hay 'dolo o imprudencia'?")
    rag_system.ask("¿Qué dice el Artículo 2 sobre las leyes penales que favorecen al reo?")

    # --- Preguntas sobre el Libro I (Artículos 10+) ---
    rag_system.ask("¿Cómo define el Artículo 10 qué 'Son delitos'?")
    rag_system.ask("¿Cuál es la diferencia entre un delito grave, menos grave y leve, según el Artículo 13?")
    rag_system.ask("¿Cuáles son los tres requisitos para la 'legítima defensa' (defensa de la persona o derechos propios) según el Artículo 20.4?")
    rag_system.ask("Mencione tres 'circunstancias agravantes' listadas en el Artículo 22.")
    rag_system.ask("¿Cuál es la diferencia entre 'autores' (Artículo 28) y 'cómplices' (Artículo 29)?")
    rag_system.ask("Según el Artículo 32, ¿cuáles son las tres clases de penas que pueden imponerse?")
    # --- END MODIFICATION ---


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
import abc
import getpass
import os
import bs4
from typing import List, Any
from dataclasses import dataclass, field

# --- Core LangChain Imports ---
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, InMemoryVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.documents import Document
from langchain_classic import hub

# --- Provider-Specific Imports ---
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# --- API Key Setup (for Groq) ---
if not os.environ.get("GROQ_API_KEY"):
    # Check if we're in a non-interactive environment
    if 'getpass' not in dir() or not callable(getpass.getpass):
        print("GROQ_API_KEY not found. Please set it as an environment variable.")
        # As a fallback for non-interactive environments
        api_key_from_input = input("Enter API key for Groq: ")
        if api_key_from_input:
            os.environ["GROQ_API_KEY"] = api_key_from_input
        else:
            print("No API key provided. Groq will not work.")
    else:
        try:
            print("GROQ_API_KEY not found in environment variables.")
            os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
        except Exception as e:
            print(f"Could not get API key: {e}. Please set it as an environment variable.")


# --- Configuration Dataclass (Restored) ---

@dataclass
class RAGConfig:
    """
    Configuration class for the RAG system to make hyperparameters configurable.
    """
    # Loader Config
    urls: List[str] = field(default_factory=lambda: ["https://lilianweng.github.io/posts/2023-06-23-agent/"])
    
    # Splitter Config
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retriever Config
    top_k: int = 4
    
    # Generator Config
    llm_model_name: str = "llama-3.1-8b-instant"
    llm_provider: str = "groq"
    llm_temperature: float = 0.7
    prompt_hub_path: str = "rlm/rag-prompt"

    # Embeddings Config
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Note: Removed persist_directory as this class is for InMemory
    
# --- Abstract Base Classes (Interfaces) ---

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

# --- Concrete Implementations ---

class ConfigurableLoader(BaseLoader):
    """
    Concrete implementation for loading web pages.
    This version is fixed to be general-purpose (no specific bs4_strainer).
    """
    def __init__(self, urls: List[str]):
        self.urls = urls
        self.loader = WebBaseLoader(
            web_paths=self.urls,
        )

    def load(self) -> List[Document]:
        print(f"Loading documents from: {self.urls}")
        try:
            docs = self.loader.load()
            print(f"Loaded {len(docs)} documents.")
            if not docs:
                print("\nWARNING: No documents were loaded. Check URLs and site scraping permissions.\n")
            return docs
        except Exception as e:
            print(f"An error occurred during loading: {e}")
            return []

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

# --- Original RAG System Class (Composition) ---

class RAG:
    """
    Main RAG system class that composes the Indexer, Retriever, and Generator.
    (This is the original RAG class from your first script).
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


# --- New RAGInterface Class (Corrected Version) ---

class RAGInterface:
    """
    A simple interface that encapsulates the logic from `main_inmemory_groq`.
    It takes a RAGConfig object and builds the RAG system on init.
    """
    def __init__(self, config: RAGConfig):
        """
        Initializes and builds the entire RAG pipeline from a config object.
        """
        print("--- Initializing RAGInterface ---")
        self.config = config
        
        # 1. Instantiate all components based on the config
        
        # Loader
        loader = ConfigurableLoader(urls=self.config.urls)
        
        # Embeddings
        print(f"Loading embedding model: {self.config.embedding_model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        
        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
        
        # LLM
        print(f"Loading LLM: {self.config.llm_model_name} from {self.config.llm_provider}")
        if self.config.llm_provider != "groq":
            print(f"Warning: This class is optimized for Groq, but provider is {self.config.llm_provider}.")
            # You could add more provider logic here if needed
            
        llm = ChatGroq(
            model=self.config.llm_model_name, 
            temperature=self.config.llm_temperature
        )
        
        # Prompt
        print(f"Pulling prompt from Hub: {self.config.prompt_hub_path}")
        prompt = hub.pull(self.config.prompt_hub_path)
        
        # Indexer (In-Memory)
        indexer = InMemoryIndexer(
            embeddings_model=embeddings, 
            text_splitter=text_splitter
        )
        
        # Retriever (In-Memory)
        retriever = InMemoryRetriever(top_k=self.config.top_k)
        
        # Generator
        generator = LangChainGenerator(llm=llm, prompt_template=prompt)
        
        # 2. Instantiate the main RAG system
        self.rag_system = RAG(
            loader=loader,
            indexer=indexer,
            retriever=retriever,
            generator=generator
        )
        
        # 3. Run the setup pipeline (load, index, create chain)
        self.rag_system.setup_pipeline()

    def ask(self, query: str) -> str:
        """
        Passes the query directly to the internal RAG system's ask method.
        """
        return self.rag_system.ask(query)


# --- Example Usage ---

if __name__ == "__main__":
    
    print("--- Testing RAGInterface with 'Código Penal' config ---")
    
    # 1. Create a configuration object
    # These are the exact parameters from your main_inmemory_groq function
    penal_code_config = RAGConfig(
        urls=[
            "https://www.boe.es/buscar/act.php?id=BOE-A-1995-25444", # BOE - Texto Consolidado
            "https://es.wikipedia.org/wiki/C%C3%B3digo_Penal_de_Espa%C3%B1a", # Wikipedia
            "https://www.conceptosjuridicos.com/codigo-penal/" # Legal concepts summary
        ],
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
        llm_model_name="llama-3.1-8b-instant",
        llm_provider="groq",
        llm_temperature=0.9,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 2. Instantiate the RAGInterface with the config object
    penal_code_bot = RAGInterface(config=penal_code_config)

    # 3. Ask questions
    # --- Preguntas sobre la Exposición de Motivos (Preamble) ---
    penal_code_bot.ask("Según la exposición de motivos, ¿por qué se considera el Código Penal como una 'Constitución negativa'?")
    
    # --- Preguntas sobre el Título Preliminar (Artículos 1-9) ---
    penal_code_bot.ask("Basado en el Artículo 5, ¿es posible que exista una pena si no hay 'dolo o imprudencia'?")

    # --- Preguntas sobre el Libro I (Artículos 10+) ---
    penal_code_bot.ask("¿Cuál es la diferencia entre un delito grave, menos grave y leve, según el Artículo 13?")

