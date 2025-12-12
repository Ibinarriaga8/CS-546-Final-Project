# rag_interface.py (Performance Optimized Wrapper)

from rag import *

import matplotlib.pyplot as plt
from utils import similarity   

_EMBEDDINGS_CACHE = {}

class RAGInterface:
    """
    Wrapper class for the RAG system designed for the RL/PPO pipeline.
    It uses a static cache to ensure the expensive embedding model is only loaded once.
    """
    
    def __init__(self, config: RAGConfig):
        print("--- Initializing RAGInterface (Full Setup) ---")
        self.config = config
        
        # 1. Instantiate Embeddings using the static cache (FIX: Runs once)
        self.embeddings = self._get_cached_embeddings(config.embedding_model_name)

        # 2. Instantiate all required components based on the current config
        
        # Text splitter (Dynamic: Depends on chunk_size/overlap)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        # LLM
        print(f"Loading LLM: {config.llm_model_name}")
        llm = ChatGroq(model=config.llm_model_name, temperature=config.llm_temperature)

        # Prompt
        prompt = config.custom_prompt_template or hub.pull(config.prompt_hub_path)

        # Loader (Used for document retrieval)
        loader = ConfigurableLoader(urls=config.urls, texts=config.texts)

        # Indexer (Requires embeddings/splitter)
        indexer = InMemoryIndexer(self.embeddings, splitter)

        # Retriever (Dynamic: Depends on top_k)
        retriever = InMemoryRetriever(top_k=config.top_k)

        # Generator
        generator = LangChainGenerator(llm, prompt)

        # 3. Instantiate the RAG Composition Class
        self.rag_system = RAG(
            loader=loader,
            indexer=indexer,
            retriever=retriever,
            generator=generator
        )

        # 4. Run the full pipeline setup (Load Docs, Index, Build Chain)
        self.rag_system.setup_pipeline()
        
        
    @staticmethod
    def _get_cached_embeddings(model_name: str) -> HuggingFaceEmbeddings:
        """
        Ensures the HuggingFaceEmbeddings model is loaded only once across 
        all instances of RAGInterface.
        """
        if model_name not in _EMBEDDINGS_CACHE:
            print(f"Loading embedding model: {model_name} (FIRST TIME)")
            _EMBEDDINGS_CACHE[model_name] = HuggingFaceEmbeddings(model_name=model_name)
        else:
            print(f"Using cached embedding model: {model_name}")
        
        return _EMBEDDINGS_CACHE[model_name]


    def ask(self, query):
        """
        Invokes the RAG system's ask method.
        """
        return self.rag_system.ask(query)



class QABatchProcessor:
    """Processes a batch of QA pairs."""
    def __init__(self, config: RAGConfig, questions_file: str,
                    output_file: str = "rag_results.txt"):
        self.rag_interface = RAGInterface(config)
        self.questions_file = questions_file
        self.output_file = output_file

    def process_batch(self):
        with open(self.questions_file, 'r') as infile, open(self.output_file, 'w') as outfile:
            for line in infile:
                question = line.strip()
                if not question: continue
                answer = self.rag_interface.ask(question)
                outfile.write(f"Q: {question}\n")
                outfile.write(f"A: {answer}\n")
                outfile.write("-" * 80 + "\n\n")



if __name__ == "__main__":
    
    config = RAGConfig(
        urls=[
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Artificial_neural_network"
        ],
        chunk_size=1000,
        chunk_overlap=200,
        top_k=4,
        llm_model_name="llama-3.1-8b-instant",
        llm_provider="groq",
        llm_temperature=0.2,
        embedding_model_name="sentence-transformers/all-mpnet-base-v2"
    )
        
    rag = RAGInterface(config)

    # Test determinism of RAG
    question = "What is Deep Learning?"
    num_runs = 15

    responses = []
    for i in range(num_runs):
        ans = rag.ask(question)
        responses.append(ans)

    base_answer = responses[0]
    similarities = []

    for i, ans in enumerate(responses):
        sim = similarity(base_answer, ans)
        similarities.append(sim)

    plt.figure()
    plt.plot(range(num_runs), similarities, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Cosine similarity with first answer")
    plt.title("RAG determinism across repeated queries")
    plt.grid(True)
    plt.savefig("rag_determinism.png")
    plt.show()