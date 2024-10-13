import re, os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer  # Add this import
from together import Together
from dotenv import load_dotenv
from pprint import pprint
import spacy
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever, EmbedchainRetriever
import faiss
from rank_bm25 import BM25Okapi
from pathlib import Path

load_dotenv()


# # Example usage
# kb = KnowledgeBase()
# text = "Barack Obama was born in Hawaii and served as the 44th President of the United States."
# tagged_text = kb.named_entity_tagging(text)
# print("Tagged Text:", tagged_text)

import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk


class QueryProcessor:
    def __init__(self):
        # Initialize the stemmer
        self.stemmer = PorterStemmer()

        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        # Load stopwords
        self.stop_words = set(stopwords.words("english"))
        self.nlp = spacy.load("en_core_web_sm")

    def process_query(self, query):
        # print(f"Original Query: {query}\n")

        # Step 1: Convert to lowercase
        query_lower = self.to_lowercase(query)
        # print(f"Step 1 - Lowercase: {query_lower}\n")

        # Step 2: Remove special characters
        query_clean = self.remove_special_characters(query_lower)
        # print(f"Step 2 - Remove Special Characters: {query_clean}\n")

        # Step 3: Tokenize (sentences and words)
        sentences, words = self.tokenize(query_clean)
        # print(f"Step 3 - Sentence Tokens: {sentences}")
        # print(f"Step 3 - Word Tokens: {words}\n")

        # Step 4: Stemming
        stemmed_words = [self.stemmer.stem(word) for word in words]
        # print(f"Step 4 - Stemmed Words: {stemmed_words}\n")

        # Step 5: Lemmatization
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        # print(f"Step 5 - Lemmatized Words: {lemmatized_words}\n")

        # Step 6: Remove Stopwords
        filtered_words = [
            word for word in lemmatized_words if word not in self.stop_words
        ]
        # print(f"Step 6 - Without Stopwords: {filtered_words}\n")

        # POS tagging and NER
        pos_tags, entities = self.pos_and_ner(query_clean)
        # print(f"POS Tags: {pos_tags}")
        # print(f"Named Entities: {entities}\n")

        # Generate context-based list of 3-word strings
        context_based_chunks = self.generate_3_word_chunks(
            filtered_words, pos_tags, entities
        )
        print(f"Context-Based Three-Word Strings: {context_based_chunks}\n")

        return context_based_chunks

    def to_lowercase(self, text):
        return text.lower()

    def remove_special_characters(self, text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)

    def tokenize(self, text):
        sentence_tokens = sent_tokenize(text)
        word_tokens = word_tokenize(text)
        return sentence_tokens, word_tokens

    def pos_and_ner(self, text):
        doc = self.nlp(text)

        # POS tagging
        pos_tags = [(token.text, token.pos_) for token in doc]

        # Named Entity Recognition (NER)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        return pos_tags, entities

    def generate_3_word_chunks(self, words, pos_tags, entities):
        """
        Generate contextually relevant 3-word chunks using POS tags and NER.
        """
        chunks = []
        current_chunk = []

        for word, pos in pos_tags:
            if pos in {
                "NOUN",
                "PROPN",
                "VERB",
                "ADJ",
            }:  # Select nouns, proper nouns, verbs, adjectives
                current_chunk.append(word)

            if len(current_chunk) == 3:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        # Handle the case where fewer than 3 words are left in the chunk
        if len(current_chunk) > 0:
            chunks.append(" ".join(current_chunk))

        # Ensure entities are part of the output for better context
        entity_chunks = [
            " ".join(entity[0] for entity in entities if entity[0] in words)
        ]
        if entity_chunks and not chunks:
            return entity_chunks

        return chunks or entity_chunks


class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)


class KnowledgeBase:
    def __init__(self):
        api_key = os.getenv("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        self.client = Together(api_key=api_key)

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformerWrapper(self.model)  # Use the wrapper
        self.nlp = spacy.load("en_core_web_sm")

        self.storage_dir = "VectorDB"
        self.documents = []
        self.create_directory_structure()
        self.title_index = {}
        self.summary_index = {}
        self.text_index = {}
        self.title_documents = {}
        self.summary_documents = {}
        self.text_documents = {}

        # try:
        self.load_knowledge_base()
        # except Exception as e:
        # print("ERROR :: ", e)
        self.chunks = []

    def create_directory_structure(self):

        # subdirs = ["documents"]
        # , "embeddings", "metadata", "index"]
        for subdir in self.documents:
            os.makedirs(Path(self.storage_dir) / subdir, exist_ok=True)
        print("Directory structure created successfully.")

    def read_index(self, path):
        if os.path.exists(os.path.join(self.storage_dir, path)):
            return faiss.read_index(os.path.join(self.storage_dir, path))
        return None

    def queryLLM(self, prompt):
        try:
            stream = self.client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            response_text = ""
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        if choice.delta.content is not None:
                            response_text += choice.delta.content
                elif hasattr(chunk, "content"):
                    response_text += chunk.content

            return response_text.strip()
        except Exception as e:
            print(f"Error in AI generation: {str(e)}")
            return (
                "I'm sorry, but I encountered an error while processing your request."
            )

    def get_embeddings(self, text_list):
        embeddings = self.model.encode(text_list)  # Use the initialized model
        return embeddings

    def split_sentences(self, text):
        return re.split(r"(?<=[.?!])\s+", text)

    def combine_sentences(self, sentences, buffer_size=1):
        combined_sentences = []
        for i in range(len(sentences)):
            combined_sentence = ""

            # Add sentences before the current one
            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j] + " "

            # Add the current sentence
            combined_sentence += sentences[i]

            # Add sentences after the current one
            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += " " + sentences[j]

            combined_sentences.append(combined_sentence.strip())

        return combined_sentences

    def calculate_cosine_distances(self, embeddings):
        distances = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def plot_distances(self, distances, threshold):
        plt.plot(distances)
        plt.axhline(y=threshold, color="r", linestyle="-")
        plt.title("Semantic Chunking Based on Embedding Breakpoints")
        plt.xlabel("Sentence Position")
        plt.ylabel("Cosine Distance")
        plt.show()

    def semantic_chunking(self, text, fname, buffer_size=1, percentile_threshold=75):
        # Step 1: Split text into sentences
        sentences = self.split_sentences(text)

        # Step 2: Combine sentences with buffer
        combined_sentences = self.combine_sentences(sentences, buffer_size)

        # Step 3: Get embeddings for the combined sentences
        embeddings = self.get_embeddings(combined_sentences)

        # Step 4: Calculate cosine distances between sequential embeddings
        distances = self.calculate_cosine_distances(embeddings)

        # Step 5: Calculate the distance threshold for breakpoints
        breakpoint_distance_threshold = np.percentile(distances, percentile_threshold)

        # Step 6: Identify the sentence indices where breaks should occur
        indices_above_threshold = [
            i
            for i, distance in enumerate(distances)
            if distance > breakpoint_distance_threshold
        ]

        # Step 7: Plot the distances and threshold
        self.plot_distances(distances, breakpoint_distance_threshold)

        # Step 8: Group sentences into chunks based on breakpoints

        start_index = 0
        uniq_index = 0
        for index in indices_above_threshold:
            chunk = " ".join(sentences[start_index : index + 1])
            self.chunks.append(
                {
                    "title": self.get_new_chunk_title(chunk),
                    "chunk": self.named_entity_tagging(chunk),
                    "summary": self.get_new_chunk_summary(chunk),
                    "index": uniq_index,
                    "lines": (start_index, index + 1),
                    "document Name": fname,
                }
            )
            start_index = index + 1
            uniq_index += 1

        # Append the last chunk if any sentences remain
        if start_index < len(sentences):
            chunk = " ".join(sentences[start_index:])
            self.chunks.append(
                {
                    "title": self.get_new_chunk_title(chunk),
                    "chunk": self.named_entity_tagging(chunk),
                    "summary": self.get_new_chunk_summary(chunk),
                    "index": uniq_index,
                    "lines": (start_index, len(sentences)),
                    "document Name": fname,
                }
            )

        self.save_chunks_separate_indexes()

        return self.chunks

    def get_new_chunk_summary(self, proposition):
        try:
            stream = self.client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                        You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                        A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                        You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                        Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                        Or month, generalize it to "date and times", etc.

                        Example:
                        Input: Proposition: Greg likes to eat pizza
                        Output: This chunk contains information about the types of food Greg likes to eat.

                        Only respond with the new chunk summary, nothing else.
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"Determine the summary of the new chunk that this proposition will go into:\n{proposition}",
                    },
                ],
                stream=True,
            )

            response_text = ""
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        if choice.delta.content is not None:
                            response_text += choice.delta.content
                elif hasattr(chunk, "content"):
                    response_text += chunk.content

            return response_text.strip()
        except Exception as e:
            print(f"Error in AI generation: {str(e)}")
            return ""

    def get_new_chunk_title(self, proposition):
        try:
            stream = self.client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[
                    {
                        "role": "system",
                        "content": """
                            You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
                            You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                            A good chunk title is brief but encompasses what the chunk is about

                            You will be given a summary of a chunk which needs a title

                            Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                            Or month, generalize it to "date and times".

                            Example:
                            Input: Summary: This chunk is about dates and times that the author talks about
                            Output: Date & Times

                            Only respond with the new chunk title, nothing else.
                            """,
                    },
                    {
                        "role": "user",
                        "content": f"Determine the title of the chunk that this summary belongs to:\n{proposition}",
                    },
                ],
                stream=True,
            )

            response_text = ""
            for chunk in stream:
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                        if choice.delta.content is not None:
                            response_text += choice.delta.content
                elif hasattr(chunk, "content"):
                    response_text += chunk.content

            return response_text.strip()
        except Exception as e:
            print(f"Error in AI generation: {str(e)}")
            return ""

    def named_entity_tagging(self, text):
        """
        Accepts a string as input, applies NER on the text using spaCy,
        and modifies the chunk with entity tags.
        """
        doc = self.nlp(text)  # Apply spaCy's NLP pipeline to the input text
        modified_text = text

        # Loop through the recognized entities and modify the text to tag entities
        for ent in doc.ents:
            full_label = spacy.explain(ent.label_)
            if full_label:
                entity_tag = f"[{full_label}: {ent.text}]"
            else:
                entity_tag = f"[{ent.label_}: {ent.text}]"  # Format entity tag
            # Replace the original text with the tagged version
            modified_text = modified_text.replace(ent.text, entity_tag)

        return modified_text

    def modify_chunk_with_ner(self, chunk_index):
        """
        Modifies a specific chunk based on Named Entity Recognition (NER)
        and updates the chunk in the knowledge base.
        """
        if 0 <= chunk_index < len(self.chunks):
            chunk = self.chunks[chunk_index]["chunk"]
            modified_chunk = self.named_entity_tagging(chunk)
            self.chunks[chunk_index]["chunk"] = modified_chunk
            return self.chunks[chunk_index]
        else:
            return f"Invalid chunk index: {chunk_index}"

    # [Include all other methods from previous examples]

    def save_chunks_separate_indexes(self):
        """
        Saves the chunks' titles, summaries, and texts in separate indexes.
        """
        groupedChunk = {}
        for chunk in self.chunks:
            doc = chunk["document Name"]
            if doc not in groupedChunk:
                groupedChunk[doc] = []
            groupedChunk[doc].append(chunk)

        self.title_documents = {}
        self.summary_documents = {}
        self.text_documents = {}

        for doc in groupedChunk:

            title_texts = [chunk.get("title", "") for chunk in groupedChunk[doc]]
            summary_texts = [chunk.get("summary", "") for chunk in groupedChunk[doc]]
            chunk_texts = [chunk.get("chunk", "") for chunk in groupedChunk[doc]]

            if doc not in self.title_index:
                self.title_index[doc] = None
            if doc not in self.summary_index:
                self.summary_index[doc] = None
            if doc not in self.text_index:
                self.text_index[doc] = None

            # Create FAISS vector stores for title, summary, and chunk texts
            self.title_index[doc] = FAISS.from_texts(title_texts, self.embedding_model)
            self.summary_index[doc] = FAISS.from_texts(
                summary_texts, self.embedding_model
            )
            self.text_index[doc] = FAISS.from_texts(chunk_texts, self.embedding_model)

            if doc not in self.title_documents:
                self.title_documents[doc] = []
            if doc not in self.summary_documents:
                self.summary_documents[doc] = []
            if doc not in self.text_documents:
                self.text_documents[doc] = []

            self.title_documents[doc] = title_texts
            self.summary_documents[doc] = summary_texts
            self.text_documents[doc] = chunk_texts

            os.makedirs(Path(self.storage_dir) / doc, exist_ok=True)

            # os.makedirs(Path(self.storage_dir) / fname, exist_ok=True)
            print(f"Directory structure { doc } created successfully.")

            faiss.write_index(
                self.title_index[doc].index,
                os.path.join(self.storage_dir, doc, "title_index.faiss"),
            )
            faiss.write_index(
                self.summary_index[doc].index,
                os.path.join(self.storage_dir, doc, "summary_index.faiss"),
            )
            faiss.write_index(
                self.text_index[doc].index,
                os.path.join(self.storage_dir, doc, "text_index.faiss"),
            )
            # Save document texts
            with open(os.path.join(self.storage_dir, doc, "title_texts.txt"), "w") as f:
                f.write("\n".join(self.title_documents))
            with open(
                os.path.join(self.storage_dir, doc, "summary_texts.txt"), "w"
            ) as f:
                f.write("\n".join(self.summary_documents))
            with open(os.path.join(self.storage_dir, doc, "chunk_texts.txt"), "w") as f:
                f.write("\n".join(self.text_documents))
            print(
                f"Folder : {doc} Chunks have been saved in FAISS indexes for title, summary, and text."
            )

    def selectedIndexSearch(self, fnames, query):
        print(fnames)
        self.documents = []
        self.title_index = {}
        self.summary_index = {}
        self.text_index = {}
        self.title_documents = {}
        self.summary_documents = {}
        self.text_documents = {}
        for doc in os.listdir(self.storage_dir):
            if doc in fnames:
                if doc not in self.title_index:
                    self.title_index[doc] = None
                if doc not in self.summary_index:
                    self.summary_index[doc] = None
                if doc not in self.text_index:
                    self.text_index[doc] = None
                if doc not in self.title_documents:
                    self.title_documents[doc] = None
                if doc not in self.summary_documents:
                    self.summary_documents[doc] = None
                if doc not in self.text_documents:
                    self.text_documents[doc] = None

                try:
                    self.title_index[doc] = self.read_index(f"{doc}/title_index.faiss")
                    self.summary_index[doc] = self.read_index(
                        f"{doc}/summary_index.faiss"
                    )
                    self.text_index[doc] = self.read_index(f"{doc}/text_index.faiss")
                    # print(self.text_index)
                    with open(
                        os.path.join(self.storage_dir, doc, "title_texts.txt"), "r"
                    ) as f:
                        self.title_documents[doc] = f.read().splitlines()
                    with open(
                        os.path.join(self.storage_dir, doc, "summary_texts.txt"), "r"
                    ) as f:
                        self.summary_documents[doc] = f.read().splitlines()
                    with open(
                        os.path.join(self.storage_dir, doc, "chunk_texts.txt"), "r"
                    ) as f:
                        self.text_documents[doc] = f.read().splitlines()

                    print("FAISS indexes and documents have been loaded from storage.")
                    self.documents.append(doc)
                except Exception as e:
                    print("Error > : ", e)

            self.documents.append(doc)
        return self.retrieve_and_rerank(query)
        # pass

    def load_knowledge_base(self):
        """
        Loads the FAISS indexes and documents from storage.
        """
        self.documents = []
        # for doc in os.listdir(self.storage_dir):
        # print(self.title_index, self.read_index(f"{doc}/title_index.faiss"))
        for doc in os.listdir(self.storage_dir):
            if doc not in self.title_index:
                self.title_index[doc] = None
            if doc not in self.summary_index:
                self.summary_index[doc] = None
            if doc not in self.text_index:
                self.text_index[doc] = None
            if doc not in self.title_documents:
                self.title_documents[doc] = None
            if doc not in self.summary_documents:
                self.summary_documents[doc] = None
            if doc not in self.text_documents:
                self.text_documents[doc] = None

            try:
                self.title_index[doc] = self.read_index(f"{doc}/title_index.faiss")
                self.summary_index[doc] = self.read_index(f"{doc}/summary_index.faiss")
                self.text_index[doc] = self.read_index(f"{doc}/text_index.faiss")
                # print(self.text_index)
                with open(
                    os.path.join(self.storage_dir, doc, "title_texts.txt"), "r"
                ) as f:
                    self.title_documents[doc] = f.read().splitlines()
                with open(
                    os.path.join(self.storage_dir, doc, "summary_texts.txt"), "r"
                ) as f:
                    self.summary_documents[doc] = f.read().splitlines()
                with open(
                    os.path.join(self.storage_dir, doc, "chunk_texts.txt"), "r"
                ) as f:
                    self.text_documents[doc] = f.read().splitlines()

                print("FAISS indexes and documents have been loaded from storage.")
                self.documents.append(doc)
            except Exception as e:
                print("Error > : ", e)

    def add_to_knowledge_base(self, text, fname):

        self.semantic_chunking(text, fname)

    # def retrieve_and_rerank(self, query):
    #     """
    #     Searches through all indexes (title, summary, text) and reranks the results using SentenceTransformer embeddings and FAISS.
    #     """
    #     # Step 1: Generate the query embedding using SentenceTransformer
    #     query_embedding = self.model.encode(query, convert_to_tensor=False)

    #     # Step 2: Search through the title, summary, and text FAISS indexes
    #     title_results = self.title_index.similarity_search_by_vector(query_embedding)
    #     summary_results = self.summary_index.similarity_search_by_vector(
    #         query_embedding
    #     )
    #     text_results = self.text_index.similarity_search_by_vector(query_embedding)

    #     # Step 3: Combine the results from all indexes
    #     combined_results = title_results + summary_results + text_results

    #     # Step 4: Rerank the combined results based on their similarity score (assuming 'score' is available)
    #     reranked_results = sorted(
    #         combined_results, key=lambda x: x["score"], reverse=True
    #     )

    #     return reranked_results

    def embedding_retrieve(self, query, index, model):
        query_embedding = model.encode(
            query
        )  # Assuming you're using SentenceTransformer for embeddings
        retriever = EmbedchainRetriever(vectorstore=index)
        results = retriever.get_relevant_documents(query_embedding)
        return results

    def faiss_retrieve(self, query, texts, index, k=5):
        """
        Retrieves documents from the specified FAISS index.
        """
        f_distances = {}
        f_indices = {}
        for doc in texts:
            # query_embedding = self.model.encode(query)
            distances, indices = index[doc].search(query, k)
            f_distances[doc] = distances
            f_indices[doc] = indices

        # return [
        #     {"text": texts[idx], "distance": distances[0][i]}
        #     for i, idx in enumerate(indices[0])

        # ]
        results = {}
        for doc in f_indices:
            results[doc] = []
            for i, idx in enumerate(f_indices[doc][0]):
                results[doc].append(
                    {
                        "text": texts[doc][idx],
                        "distance": f_distances[doc][0][i],
                    }
                )

        # print(results)

        # def similarity_search(query, index, texts, k=5):
        #     distances, indices = index.search(query, k)
        #     results = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
        return results

    def bm25_search(self, query, texts, k=5):
        # Tokenize the texts and the query
        results = {}
        tokenized_query = query.split(" ")
        for doc in texts:

            tokenized_texts = [docs.split() for docs in texts[doc]]
            bm25 = BM25Okapi(tokenized_texts)

            # BM25 search
            bm25_scores = bm25.get_scores(tokenized_query)
            top_n_indices = bm25.get_top_n(tokenized_query, range(len(texts[doc])), n=k)

            results[doc] = [
                {"text": texts[doc][i], "distance": bm25_scores[i]}
                for i in top_n_indices
            ]

        # print(results)
        return results

    def remove_duplicates(self, dictionaries):
        unique_texts = set()
        unique_dictionaries = []

        for dictionary in dictionaries:
            text = dictionary["text"]
            if text not in unique_texts:
                unique_texts.add(text)
                unique_dictionaries.append(dictionary)

        return unique_dictionaries

    def rerank_results(self, combined_results, top_n, threshold=0.5):
        filtered_results = []
        for result in combined_results:
            # print(result)
            for doc in result:
                for chunk in result[doc]:

                    if chunk["distance"] > threshold:
                        chunk["document Name"] = doc
                        filtered_results.append(chunk)

        # filtered_results = [
        #     results for results in combined_results if results["distance"] > threshold
        # ]
        # Sort results by score (assuming each result has a 'score' field)

        reranked_results = self.remove_duplicates(
            filtered_results[:top_n]
            if len(filtered_results) > top_n
            else filtered_results
        )
        result = sorted(filtered_results, key=lambda x: x["distance"], reverse=True)
        # Return the top N results or all results if fewer than N
        return result

    def retrieve_and_rerank(self, queries, top_n=5):
        results = []
        for query in queries:
            text_query = query
            query = self.model.encode([query]).astype(np.float32)
            # Retrieve results from different indexes using each retriever
            # title_bm25_results = self.bm25_retrieve(query, self.title_index)
            # summary_embedding_results = self.embedding_retrieve(
            #     query, self.summary_index, self.model
            # )

            # print(self.faiss_retrieve(query, self.text_documents, self.text_index))

            faiss_results = self.faiss_retrieve(
                query, self.text_documents, self.text_index
            )
            results.append(faiss_results)
            bm25_result = self.bm25_search(text_query, self.text_documents)
            results.append(bm25_result)

        # print(results)
        # Combine and rerank
        final_results = self.rerank_results(
            # title_bm25_results, summary_embedding_results,
            results,
            top_n,
        )

        return final_results

    def list_documents(self):
        return self.documents


if __name__ == "__main__":

    ai = KnowledgeBase()

    # # Example usage:
    text = """
    India's rich cultural heritage is a treasure trove of diverse traditions, customs, and practices that have been shaped over thousands of years. From the ancient Indus Valley Civilization to the present day, India has been a melting pot of various cultures, each contributing to the country's unique identity. The country's cultural landscape is characterized by a blend of Hinduism, Buddhism, Jainism, and Islam, with each religion having its own distinct festivals, rituals, and traditions. The vibrant colors, intricate patterns, and ornate designs of Indian art and architecture are a testament to the country's rich cultural heritage. The majestic Taj Mahal, the stunning Red Fort, and the intricate temples of Khajuraho are just a few examples of India's architectural wonders. India's cultural heritage is also reflected in its music, dance, and literature, with the likes of Rabindranath Tagore, Mahatma Gandhi, and Ravi Shankar being celebrated worldwide. The country's diverse cultural heritage is a reflection of its history, philosophy, and values, which continue to inspire and influence people around the world.India is home to a staggering array of wildlife, with over 1,200 species of birds, 500 species of fish, and 300 species of mammals. The country's diverse geography, ranging from the snow-capped Himalayas to the scorching deserts of Rajasthan, supports a wide range of ecosystems, each with its unique set of flora and fauna. The Sundarbans, the largest mangrove forest in the world, is home to the majestic Bengal tiger, while the Gir Forest National Park is the only habitat of the endangered Asiatic lion. India's wildlife is also characterized by its rich variety of birdlife, with species like the peacock, the parrot, and the eagle being iconic symbols of the country. The country's wetlands, including the famous Keoladeo National Park, are home to a wide range of waterbirds, including the majestic flamingo and the elegant pelican. India's wildlife is not only a source of national pride but also a vital component of the country's ecosystem, providing essential services like pollination, pest control, and climate regulation. The conservation of India's wildlife is a priority, with efforts being made to protect and preserve the country's natural heritage for future generations.India is a country that loves to celebrate, with a wide range of festivals and events taking place throughout the year. From the colorful Holi festival of colors to the majestic Diwali festival of lights, India's festivals are a reflection of the country's rich cultural heritage. The Navratri festival, celebrated over nine days, is a celebration of music, dance, and devotion, with people from all over the country coming together to dance and sing. The Dussehra festival, which marks the victory of good over evil, is celebrated with great fervor, with effigies of Ravana being burned in public. India's festivals are also a time for family and friends to come together, with delicious food, sweet treats, and warm hospitality being an integral part of the celebrations. The country's festivals are not just a source of joy and entertainment but also a way of connecting with one's heritage and culture. Whether it's the majestic Kumbh Mela, the vibrant Ganesh Chaturthi, or the joyous Eid-al-Fitr, India's festivals are a celebration of life, love, and community.India has a rich and diverse history that spans over 5,000 years, with the Indus Valley Civilization being one of the oldest civilizations in the world. The country's history is marked by a series of empires, including the Mauryan, the Gupta, and the Mughal, each leaving its own unique legacy. The ancient Indian kingdoms of Magadha, Kalinga, and Ujjain were known for their rich cultural heritage, with the likes of Chanakya and Aryabhata making significant contributions to the field of politics and mathematics. India's history is also marked by its struggle for independence, with the likes of Mahatma Gandhi, Jawaharlal Nehru, and Subhas Chandra Bose playing a crucial role in the country's freedom movement. The country's history is also characterized by its rich literary and artistic heritage, with the likes of Kalidasa, Rabindranath Tagore, and Ravi Shankar being celebrated worldwide. India's history is a reflection of its people, their struggles, and their triumphs, and it continues to inspire and influence people around the world.Indian cuisine is a reflection of the country's rich cultural heritage, with a wide range of dishes and flavors being enjoyed across the country. From the spicy curries of the south to the rich biryanis of the north, Indian cuisine is a fusion of different flavors, textures, and aromas. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of crops, including rice, wheat, and lentils, which are staples of the Indian diet. Indian cuisine is also characterized by its rich use of spices, with turmeric, cumin, coriander, and chili peppers being some of the most commonly used spices. The country's cuisine is not just a source of nourishment but also a way of connecting with one's heritage and culture. Whether it's the delicious biryani of Hyderabadi, the spicy vada pav of Mumbai, or the sweet jalebi of Jaipur, Indian cuisine is a reflection of the country's rich culinary heritage.Indian fashion is a reflection of the country's rich cultural heritage, with a wide range of styles, fabrics, and designs being enjoyed across the country. From the traditional sarees of the south to the elegant salwar kameez of the north, Indian fashion is a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of crops, including cotton, silk, and wool, which are used to create a wide range of fabrics. Indian fashion is also characterized by its rich use of colors, with bright hues, intricate patterns, and ornate designs being a hallmark of Indian fashion. The country's fashion industry is not just a source of employment but also a way of connecting with one's heritage and culture. Whether it's the elegant lehengas of Bollywood, the stylish kurtas of the streets, or the traditional angavastram of the temples, Indian fashion is a reflection of the country's rich cultural heritage.Indian music is a reflection of the country's rich cultural heritage, with a wide range of genres, styles, and instruments being enjoyed across the country. From the classical music of the north to the folk music of the south, Indian music is a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of musical instruments, including the sitar, the tabla, and the tanpura. Indian music is also characterized by its rich use of melodies, with complex ragas and intricate compositions being a hallmark of Indian music. The country's music industry is not just a source of entertainment but also a way of connecting with one's heritage and culture. Whether it's the classical music of Ravi Shankar, the folk music of Lata Mangeshkar, or the pop music of A.R. Rahman, Indian music is a reflection of the country's rich cultural heritage.Indian literature is a reflection of the country's rich cultural heritage, with a wide range of genres, styles, and authors being enjoyed across the country. From the ancient epics of the Ramayana and the Mahabharata to the modern novels of Salman Rushdie and Arundhati Roy, Indian literature is a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of literary traditions, including the Sanskrit, the Tamil, and the Urdu. Indian literature is also characterized by its rich use of language, with complex metaphors, intricate symbolism, and powerful imagery being a hallmark of Indian literature. The country's literary industry is not just a source of entertainment but also a way of connecting with one's heritage and culture. Whether it's the classical poetry of Kalidasa, the modern fiction of Rohinton Mistry, or the children's literature of Ruskin Bond, Indian literature is a reflection of the country's rich cultural heritage.Indian arts and crafts are a reflection of the country's rich cultural heritage, with a wide range of traditional crafts, such as textiles, pottery, and metalwork, being enjoyed across the country. From the intricate embroidery of the Kashmiri phiran to the colorful block printing of the Rajasthani sarees, Indian arts and crafts are a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of natural materials, such as cotton, silk, and wool, which are used to create a wide range of crafts. Indian arts and crafts are also characterized by their rich use of colors, with bright hues, intricate patterns, and ornate designs being a hallmark of Indian crafts. The country's artisans are not just skilled craftsmen but also masters of their trade, with each craft being passed down through generations. Whether it's the traditional woodcarvings of the Andhra Pradesh, the beautiful paintings of the Madhubani, or the exquisite jewelry of the Rajasthan, Indian arts and crafts are a reflection of the country's rich cultural heritage.Indian sports are a reflection of the country's rich cultural heritage, with a wide range of traditional and modern sports being enjoyed across the country. From the ancient game of chess to the modern game of cricket, Indian sports are a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of outdoor activities, such as trekking, mountaineering, and water sports. Indian sports are also characterized by their rich use of physical fitness, with athletes from across the country competing in international events. The country's sports industry is not just a source of entertainment but also a way of connecting with one's heritage and culture. Whether it's the traditional game of kabaddi, the modern game of badminton, or the popular game of cricket, Indian sports are a reflection of the country's rich cultural heritage.Indian science and technology have a rich history, with ancient Indians making significant contributions to the field of mathematics, astronomy, and medicine. From the ancient Indian concept of zero to the modern Indian space program, Indian science and technology have been a driving force behind the country's progress. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of natural resources, such as coal, iron, and copper, which are used to create a wide range of technologies. Indian science and technology are also characterized by their rich use of innovation, with scientists and engineers from across the country working on cutting-edge projects. The country's scientific community is not just a source of knowledge but also a way of connecting with one's heritage and culture. Whether it's the ancient Indian invention of the decimal system, the modern Indian development of the atomic bomb, or the Indian space program's Mars Orbiter Mission, Indian science and technology are a reflection of the country's rich cultural heritage.Indian street food is a reflection of the country's rich cultural heritage, with a wide range of traditional and modern dishes being enjoyed across the country. From the spicy chaat of the north to the sweet jalebi of the south, Indian street food is a fusion of different flavors, textures, and aromas. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of crops, including rice, wheat, and lentils, which are staples of Indian street food. Indian street food is also characterized by its rich use of spices, with turmeric, cumin, coriander, and chili peppers being some of the most commonly used spices. The country's street food vendors are not just skilled cooks but also masters of their trade, with each dish being passed down through generations. Whether it's the popular vada pav of Mumbai, the spicy samosa of Delhi, or the sweet falooda of Kolkata, Indian street food is a reflection of the country's rich cultural heritage.Indian folk music is a reflection of the country's rich cultural heritage, with a wide range of traditional and modern genres being enjoyed across the country. From the ancient folk songs of the north to the modern folk music of the south, Indian folk music is a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of musical instruments, including the sitar, the tabla, and the tanpura. Indian folk music is also characterized by its rich use of melodies, with complex ragas and intricate compositions being a hallmark of Indian folk music. The country's folk musicians are not just skilled singers but also masters of their trade, with each song being passed down through generations. Whether it's the popular folk song of the Bauls of Bengal, the traditional folk music of the Gond of Madhya Pradesh, or the modern folk music of the Punjabi, Indian folk music is a reflection of the country's rich cultural heritage.Indian textiles are a reflection of the country's rich cultural heritage, with a wide range of traditional and modern fabrics being enjoyed across the country. From the intricate embroidery of the Kashmiri phiran to the colorful block printing of the Rajasthani sarees, Indian textiles are a fusion of different styles, with each region having its own unique twist. The country's diverse geography, with its varied climate, soil, and water, supports a wide range of natural materials, such as cotton, silk, and wool, which are used to create a wide range of textiles. Indian textiles are also characterized by their rich use of colors, with bright hues, intricate patterns, and ornate designs being a hallmark of Indian textiles. The country's textile artisans are not just skilled craftsmen but also masters of their trade, with each fabric being passed down through generations. Whether it's the traditional cotton fabrics of the Andhra Pradesh, the beautiful silk fabrics of the Madhya Pradesh, or the exquisite woolen fabrics of the Jammu and Kashmir, Indian textiles are a reflection of the country's rich cultural heritage.
    """
    # chunks = ai.semantic_chunking(text)

    # pprint(chunks)

    qp = QueryProcessor()
    # ai.add_to_knowledge_base(text=text, fname="USA.txt")

    pprint(
        ai.retrieve_and_rerank(qp.process_query("tell me what you know about India"))
    )
