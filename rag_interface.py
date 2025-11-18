"""
This file is responsible for defining a RAG interface
that simplifies the implementation of RAG systems
"""


from rag import *

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



class QABatchProcessor:
    """
    This class is responsible for processing a batch of QA pairs.
    """
    def __init__(self, config: RAGConfig, questions_file: str,
                    output_file: str = "rag_results.txt"):
        
        self.rag_interface = RAGInterface(config)
        self.questions_file = questions_file
        self.output_file = output_file

    def process_batch(self):
        """
        Processes the batch of questions from the input file
        and writes the results to the output text file.
        """
        with open(self.questions_file, 'r') as infile, open(self.output_file, 'w') as outfile:
            for line in infile:
                question = line.strip()
                if not question:
                    continue
                    
                answer = self.rag_interface.ask(question)
                
                outfile.write(f"Q: {question}\n")
                outfile.write(f"A: {answer}\n")
                outfile.write("-" * 80 + "\n\n")


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
        llm_temperature=0.2,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 2. Initialize the QA Batch Processor
    qa_processor = QABatchProcessor(
        config=penal_code_config,
        questions_file="questions.txt",
    )

    # 3. Process the batch
    qa_processor.process_batch()