import os
import shutil
import pickle
import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from transformers import pipeline, BartTokenizer
from together import Together
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from gensim.models import KeyedVectors
from collections import Counter

# Download NLTK data (you may want to do this in a separate setup script)
nltk.download("punkt", quiet=True)
load_dotenv()


class RAGKnowledgeBase:
    def __init__(self, base_dir="knowledge_base"):
        self.base_dir = Path(base_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunk_size = 100  # Very small chunks
        self.overlap = 20  # Minimal overlap
        self.max_chunks_per_doc = 5  # Strict limit on chunks per document
        self.max_workers = min(4, os.cpu_count() or 1)  # Limit concurrent processing

        # Initialize Together client
        api_key = os.getenv("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        self.together_client = Together(api_key=api_key)

        self.index = None
        self.bm25 = None
        self.documents = []
        self.filenames = []
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.rerank_model = AutoModel.from_pretrained(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.contextualized_chunks = []
        self.embeddings = None
        self.embedding_cache = {}
        self.create_directory_structure()
        self.word_to_index = self.load_common_words()
        self.load_knowledge_base()

    def create_directory_structure(self):
        subdirs = ["documents", "embeddings", "metadata", "index"]
        for subdir in subdirs:
            os.makedirs(self.base_dir / subdir, exist_ok=True)
        print("Directory structure created successfully.")

    def load_knowledge_base(self):
        print("Starting to load knowledge base...")
        self.load_documents()
        if self.documents:
            self.chunk_and_contextualize_documents()
            self.create_faiss_index()
            self.create_bm25_index()
            self.save_indexes()
        print("Knowledge base loading complete.")

    def process_and_store_document(self, file_path, category):
        category_dir = self.base_dir / "documents" / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = category_dir / Path(file_path).name
        shutil.copy2(file_path, destination)
        print(f"Document stored in {destination}")

        with open(file_path, "r") as f:
            content = f.read()
        self.documents.append(content)
        print(f"Document added to knowledge base: {destination.name}")
        self.filenames.append(destination.name)

    def generate_contextual_embedding(self, document: str, chunk: str) -> np.ndarray:
        # Generate a unique key for the document-chunk pair
        key = hashlib.md5((document[:1000] + chunk[:1000]).encode()).hexdigest()

        if key in self.embedding_cache:
            return self.embedding_cache[key]

        # Extract key sentences from the document
        context = self.extract_key_sentences(document, max_sentences=3)

        # Combine the context and chunk
        final_context = f"Context: {context[:500]}...\n\nRelevant Chunk: {chunk[:500]}..."  # Limit context size

        # Generate the embedding for the context
        embedding = self.model.encode(final_context)

        self.embedding_cache[key] = embedding
        return embedding

    def extract_key_sentences(self, document: str, max_sentences: int = 5) -> str:
        # Tokenize the document into sentences
        sentences = sent_tokenize(document)

        # If the document is short, return all sentences
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        # For longer documents, extract key sentences
        key_sentences = []
        key_sentences.append(sentences[0])  # First sentence

        # Extract sentences from the middle
        middle_start = len(sentences) // 4
        middle_end = 3 * len(sentences) // 4
        step = (middle_end - middle_start) // (max_sentences - 2)
        for i in range(middle_start, middle_end, step):
            if len(key_sentences) < max_sentences - 1:
                key_sentences.append(sentences[i])

        key_sentences.append(sentences[-1])  # Last sentence

        return " ".join(key_sentences)

    def generate_and_store_embeddings(self, category):
        docs_dir = self.base_dir / "documents" / category

        embeddings = []
        for doc_path in docs_dir.glob("*"):
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    document = f.read()
                chunks = self.chunk_document(document)
                for chunk in chunks:
                    embedding = self.generate_contextual_embedding(document, chunk)
                    embeddings.append(embedding)
            except UnicodeDecodeError:
                print(
                    f"Warning: Unable to read {doc_path} with UTF-8 encoding. Skipping this file."
                )

        with open(
            self.base_dir / "embeddings" / f"{category}_embeddings.pkl", "wb"
        ) as f:
            pickle.dump(embeddings, f)

        print(f"Embeddings for {category} stored successfully.")

    def chunk_document(self, document):
        words = document.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
            if len(chunks) >= self.max_chunks_per_doc:
                break
        return chunks

    def create_and_store_metadata(self, category):
        docs_dir = self.base_dir / "documents" / category

        metadata = []
        for doc_path in docs_dir.glob("*"):
            metadata.append(
                {
                    "filename": doc_path.name,
                    "path": str(doc_path),
                    "category": category,
                    "created_at": doc_path.stat().st_ctime,
                }
            )

        with open(self.base_dir / "metadata" / f"{category}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata for {category} stored successfully.")

    def chunk_and_contextualize_documents(self):
        print(f"Processing {len(self.documents)} documents...")
        self.contextualized_chunks = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for doc in self.documents:
                chunks = self.chunk_document(doc)
                futures.extend(
                    [executor.submit(self.process_chunk, chunk) for chunk in chunks]
                )

            for future in tqdm(futures, total=len(futures), desc="Processing chunks"):
                self.contextualized_chunks.append(future.result())

        print(f"Processing complete. Total chunks: {len(self.contextualized_chunks)}")

    def process_chunk(self, chunk):
        # Simple bag-of-words representation
        word_counts = Counter(chunk.lower().split())
        vec = np.zeros(len(self.word_to_index))
        for word, count in word_counts.items():
            if word in self.word_to_index:
                vec[self.word_to_index[word]] = count
        return {"text": chunk, "context": vec}

    def create_faiss_index(self):
        if not self.contextualized_chunks:
            print("No contextualized chunks available. Cannot create FAISS index.")
            return

        embeddings = [chunk["context"] for chunk in self.contextualized_chunks]
        self.embeddings = np.array(embeddings).astype("float32")
        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        print(
            f"FAISS index created with {self.index.ntotal} vectors of dimension {dimension}"
        )

    def create_bm25_index(self):
        if self.contextualized_chunks:
            tokenized_chunks = [
                word_tokenize(chunk["text"].lower())
                for chunk in self.contextualized_chunks
            ]
            self.bm25 = BM25Okapi(tokenized_chunks)
        else:
            print("No contextualized chunks available. BM25 index not created.")

    def save_indexes(self):
        index_dir = self.base_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index using pickle
        if self.index is not None and self.index.ntotal > 0:
            try:
                with open(index_dir / "faiss_index.pkl", "wb") as f:
                    pickle.dump(self.index, f)
                print("FAISS index saved successfully using pickle.")
            except Exception as e:
                print(f"Error saving FAISS index: {e}")
                print("Skipping FAISS index save.")
        else:
            print("FAISS index is empty or not initialized. Skipping save.")

        # Save contextualized chunks (for BM25 recreation)
        with open(index_dir / "contextualized_chunks.pkl", "wb") as f:
            pickle.dump(self.contextualized_chunks, f)
        print("Contextualized chunks saved successfully.")

    def load_faiss_index(self):
        index_path = self.base_dir / "index" / "faiss_index.pkl"
        if index_path.exists():
            try:
                with open(index_path, "rb") as f:
                    self.index = pickle.load(f)
                print(
                    f"FAISS index loaded successfully. Total vectors: {self.index.ntotal}"
                )
            except Exception as e:
                print(f"Error loading FAISS index: {e}")
                print("Initializing a new FAISS index.")
                self.create_faiss_index()
        else:
            print("FAISS index not found. Please initialize the knowledge base first.")

    def load_documents(self):
        docs_dir = self.base_dir / "documents"
        self.documents = []
        self.filenames = []
        self.tokenized_documents = []
        if docs_dir.exists():
            for category_dir in docs_dir.iterdir():
                if category_dir.is_dir():
                    for doc_path in category_dir.glob("*"):
                        try:
                            with open(doc_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                self.documents.append(content)
                                self.filenames.append(str(doc_path))
                                self.tokenized_documents.append(
                                    word_tokenize(content.lower())
                                )
                        except Exception as e:
                            print(f"Error loading document {doc_path}: {str(e)}")
        print(f"Loaded {len(self.documents)} documents.")

    def load_bm25_index(self):
        chunks_path = self.base_dir / "index" / "contextualized_chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self.contextualized_chunks = pickle.load(f)
            tokenized_chunks = [
                word_tokenize(chunk["text"].lower())
                for chunk in self.contextualized_chunks
            ]
            self.bm25 = BM25Okapi(tokenized_chunks)
        else:
            print(
                "Contextualized chunks not found. Please initialize the knowledge base first."
            )

    def query_knowledge_base(self, query, k=20):
        # Load indexes if not already loaded
        if self.index is None:
            self.load_faiss_index()
        if self.bm25 is None:
            self.load_bm25_index()

        print(f"Querying knowledge base with: {query}")
        print(f"Number of documents in knowledge base: {len(self.documents)}")

        if not self.documents:
            print("No documents in the knowledge base. Unable to perform search.")
            return [], []

        # Semantic search
        sem_distances, sem_indices = self.similarity_search(query, k)
        print(f"Semantic search results: {sem_indices}")
        print(f"Semantic search distances: {sem_distances}")
        sem_docs = [self.documents[i] for i in sem_indices if i < len(self.documents)]

        # BM25 search
        if self.bm25 is None:
            print("BM25 index not initialized. Skipping BM25 search.")
            bm25_docs = []
            bm25_scores = []
        else:
            bm25_results = self.contextual_bm25_search(query, k)
            bm25_indices, bm25_scores = zip(*bm25_results)
            print(f"BM25 search results: {bm25_indices}")
            print(f"BM25 search scores: {bm25_scores}")
            bm25_docs = [
                self.documents[i] for i in bm25_indices if i < len(self.documents)
            ]

        # Combine results from semantic search and BM25
        combined_docs = list(set(sem_docs + bm25_docs))

        # Rerank the combined results
        reranked_indices = self.rerank(
            query, combined_docs, k=min(5, len(combined_docs))
        )
        final_docs = [combined_docs[i] for i in reranked_indices]

        # Get metadata for final docs
        final_indices = [self.documents.index(doc) for doc in final_docs]
        retrieved_docs = self.retrieve_documents(final_indices)

        # Calculate final distances
        final_distances = []
        for doc in final_docs:
            if doc in sem_docs:
                final_distances.append(sem_distances[sem_docs.index(doc)])
            elif doc in bm25_docs:
                final_distances.append(1 / (1 + bm25_scores[bm25_docs.index(doc)]))
            else:
                final_distances.append(float("inf"))

        print("\nRetrieved Documents:")
        for i, (doc, distance) in enumerate(zip(retrieved_docs, final_distances)):
            print(f"\nDocument {i+1}:")
            print(f"Filename: {doc.get('filename', 'Unknown')}")
            print(f"Distance: {distance}")
            print("Content:")
            print(doc.get("content", "No content available"))
            print("-" * 80)

        return retrieved_docs, final_distances

    def similarity_search(self, query, k=20):
        if self.index is None:
            print(
                "FAISS index is not initialized. Unable to perform similarity search."
            )
            return [], []

        if self.index.ntotal == 0:
            print("FAISS index is empty. Unable to perform similarity search.")
            return [], []

        # Create a bag-of-words vector for the query
        query_vec = np.zeros(len(self.word_to_index))
        for word in query.lower().split():
            if word in self.word_to_index:
                query_vec[self.word_to_index[word]] += 1

        # Ensure the query vector is float32 and reshape it
        query_vec = query_vec.astype("float32").reshape(1, -1)

        D, I = self.index.search(query_vec, min(k, self.index.ntotal))
        return D[0], I[0]

    def contextual_bm25_search(self, query, k=20):
        if self.bm25 is None:
            print("BM25 index not initialized. Unable to perform BM25 search.")
            return []

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(int(i), score) for i, score in top_n]  # Ensure indices are integers

    def rerank(self, query: str, documents: List[str], k: int = 5) -> List[int]:
        inputs = self.rerank_tokenizer(
            [query] * len(documents),
            documents,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.rerank_model(**inputs)
        scores = outputs.last_hidden_state[:, 0, :].mean(dim=1)
        top_indices = torch.argsort(scores, descending=True)[:k].tolist()
        return top_indices

    def retrieve_documents(self, indices):
        try:
            retrieved_docs = []
            for i in indices:
                if 0 <= i < len(self.documents):
                    if isinstance(self.documents[i], dict):
                        retrieved_docs.append(self.documents[i])
                    else:
                        # If the document is a string, create a simple metadata dict
                        retrieved_docs.append(
                            {
                                "content": self.documents[i][:200],  # Full content
                                "filename": (
                                    self.filenames[i]
                                    if i < len(self.filenames)
                                    else f"document_{i}"
                                ),
                                "index": i,
                            }
                        )
                else:
                    print(f"Warning: Index {i} is out of range. Skipping.")

            if not retrieved_docs:
                print("No valid documents retrieved.")
                return []

            return retrieved_docs
        except Exception as e:
            print(f"Error in retrieve_documents: {str(e)}")
            return []

    def add_to_knowledge_base(self, file_path, category):
        category_dir = self.base_dir / "documents" / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = category_dir / os.path.basename(file_path)
        shutil.copy2(file_path, destination)

        with open(destination, "r", encoding="utf-8") as f:
            content = f.read()

        self.documents.append(content)
        self.filenames.append(str(destination))

        print(f"Added document: {destination}")
        print(f"Document content preview: {content[:200]}...")

        # Update indexes
        self.chunk_and_contextualize_documents()
        self.create_faiss_index()
        self.create_bm25_index()
        self.save_indexes()

    def list_documents(self):
        documents = []
        for category_dir in (self.base_dir / "documents").iterdir():
            if category_dir.is_dir():
                for doc_path in category_dir.glob("*"):
                    documents.append(f"{category_dir.name}/{doc_path.name}")
        return documents

    def add_encrypted_data_to_knowledge_base(self):
        base_dir = Path("encrypted_data")
        for case_dir in base_dir.iterdir():
            if case_dir.is_dir():
                file_path = case_dir / f"{case_dir.name}_data.json"
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        encrypted_data = f.read()

                    decrypted_data = decrypt_data(
                        encrypted_data, username="StreamlitApp"
                    )
                    data_list = json.loads(decrypted_data)

                    for data in data_list:
                        content = json.dumps(data, indent=2)
                        metadata = {
                            "source": "realtime_data",
                            "case": case_dir.name,
                            "timestamp": data.get("timestamp", "unknown_time"),
                        }

                        self.add_document(content, metadata)

    def initialize_knowledge_base(self):
        # ... (existing code) ...

        # Add encrypted real-time data
        self.add_encrypted_data_to_knowledge_base()

        # Create FAISS index
        self.create_faiss_index()

    def add_document(self, content, metadata=None):
        if metadata is None:
            metadata = {}
        if "filename" not in metadata:
            metadata["filename"] = f"document_{len(self.documents)}"

        self.documents.append({"content": content, "metadata": metadata})

        # Process only the new document
        chunks = self.chunk_document(content)
        new_contextualized_chunks = [self.process_chunk(chunk) for chunk in chunks]

        # Update indexes
        self.update_indexes(new_contextualized_chunks)

    def update_indexes(self, new_chunks):
        # Update FAISS index
        new_embeddings = np.array([chunk["context"] for chunk in new_chunks]).astype(
            "float32"
        )
        if self.index is None:
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        self.index.add(new_embeddings)

        # Update contextualized_chunks
        self.contextualized_chunks.extend(new_chunks)

    def load_common_words(self, n_words=1000):
        common_words_path = self.base_dir / "common_words.txt"
        if not common_words_path.exists():
            # Create a simple list of common words if the file doesn't exist
            common_words = [f"word{i}" for i in range(n_words)]
            with open(common_words_path, "w") as f:
                f.write("\n".join(common_words))

        with open(common_words_path, "r") as f:
            return {word.strip(): i for i, word in enumerate(f) if i < n_words}


# Example usage
if __name__ == "__main__":
    kb = RAGKnowledgeBase()
    kb.create_directory_structure()

    # Add a document to the knowledge base
    kb.add_to_knowledge_base("path/to/your/document.pdf", "category1")

    # Query the knowledge base
    results, distances = kb.query_knowledge_base("Your query here", k=5)
    for doc, distance in zip(results, distances):
        print(f"Document: {doc['filename']}, Distance: {distance}")
