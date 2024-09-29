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
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# Download NLTK data (you may want to do this in a separate setup script)
nltk.download("punkt", quiet=True)


class RAGKnowledgeBase:
    def __init__(self, base_dir="knowledge_base"):
        self.base_dir = Path(base_dir)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
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
        self.create_directory_structure()
        self.load_documents()
        self.load_faiss_index()

    def create_directory_structure(self):
        subdirs = ["documents", "embeddings", "metadata", "index"]
        for subdir in subdirs:
            os.makedirs(self.base_dir / subdir, exist_ok=True)
        print("Directory structure created successfully.")

    def process_and_store_document(self, file_path, category):
        category_dir = self.base_dir / "documents" / category
        category_dir.mkdir(parents=True, exist_ok=True)

        destination = category_dir / Path(file_path).name
        shutil.copy2(file_path, destination)
        print(f"Document stored in {destination}")

        with open(file_path, "r") as f:
            content = f.read()
        self.documents.append(content)
        self.filenames.append(destination.name)

    def generate_contextual_embedding(self, document: str, chunk: str) -> np.ndarray:
        # Extract a brief summary or the first few sentences of the document
        doc_summary = " ".join(document.split()[:50])  # First 50 words as a summary

        context = (
            f"Document Summary: {doc_summary}\n\nRelevant Chunk: {chunk}\n\nContext:"
        )
        return self.model.encode(context)

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

    def chunk_document(self, document: str, chunk_size: int = 512) -> List[str]:
        return [
            document[i : i + chunk_size] for i in range(0, len(document), chunk_size)
        ]

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

    def create_faiss_index(self):
        if not self.documents:
            print("No documents available. Cannot create FAISS index.")
            return

        embeddings = [self.model.encode(doc) for doc in self.documents]
        embeddings = np.array(embeddings).astype("float32")
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        index_dir = self.base_dir / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss_index.bin"))

        print(
            f"FAISS index created and stored successfully. Total vectors: {self.index.ntotal}"
        )

    def load_faiss_index(self):
        index_path = self.base_dir / "index" / "faiss_index.bin"
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                print(
                    f"FAISS index loaded successfully. Total vectors: {self.index.ntotal}"
                )
            except Exception as e:
                print(f"Error loading FAISS index: {str(e)}")
                self.index = None
        else:
            print("FAISS index not found. Creating a new one.")
            self.create_faiss_index()

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
        self.create_bm25_index()
        self.create_faiss_index()

    def create_bm25_index(self):
        if self.tokenized_documents:
            self.bm25 = BM25Okapi(self.tokenized_documents)
        else:
            print("No documents loaded. BM25 index not created.")

    def query_knowledge_base(self, query, k=20):
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

        print(f"Final retrieved documents: {retrieved_docs}")
        print(f"Final distances: {final_distances}")

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

        query_embedding = self.model.encode([query])[0].astype("float32")
        D, I = self.index.search(
            query_embedding.reshape(1, -1), min(k, self.index.ntotal)
        )
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
        all_metadata = []

        for metadata_file in (self.base_dir / "metadata").glob("*_metadata.json"):
            with open(metadata_file, "r") as f:
                all_metadata.extend(json.load(f))

        retrieved_docs = [all_metadata[i] for i in indices]
        return retrieved_docs

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

        self.create_bm25_index()
        self.create_faiss_index()

    def list_documents(self):
        documents = []
        for category_dir in (self.base_dir / "documents").iterdir():
            if category_dir.is_dir():
                for doc_path in category_dir.glob("*"):
                    documents.append(f"{category_dir.name}/{doc_path.name}")
        return documents


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
