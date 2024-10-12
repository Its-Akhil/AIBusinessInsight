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

load_dotenv()


# # Example usage
# kb = KnowledgeBase()
# text = "Barack Obama was born in Hawaii and served as the 44th President of the United States."
# tagged_text = kb.named_entity_tagging(text)
# print("Tagged Text:", tagged_text)


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
        # Replace the title, summary, and text indexes with VectorStoreIndex
        # self.index = VectorStoreIndex()
        # Initialize FAISS vector stores for title, summary, and text

        self.storage_dir = "VectorDB"
        self.title_index = None
        self.summary_index = None
        self.text_index = None
        self.title_documents = None
        self.summary_documents = None
        self.text_documents = None
        try:
            self.load_chunks_separate_indexes()
        except Exception as e:
            print("ERROR : ", e)
        self.chunks = []

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

    def semantic_chunking(self, text, buffer_size=1, percentile_threshold=80):
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
        title_texts = [chunk.get("title", "") for chunk in self.chunks]
        summary_texts = [chunk.get("summary", "") for chunk in self.chunks]
        chunk_texts = [chunk.get("chunk", "") for chunk in self.chunks]

        # Create FAISS vector stores for title, summary, and chunk texts
        self.title_index = FAISS.from_texts(title_texts, self.embedding_model)
        self.summary_index = FAISS.from_texts(summary_texts, self.embedding_model)
        self.text_index = FAISS.from_texts(chunk_texts, self.embedding_model)

        # Store the documents separately
        self.title_documents = title_texts
        self.summary_documents = summary_texts
        self.text_documents = chunk_texts

        faiss.write_index(
            self.title_index.index, os.path.join(self.storage_dir, "title_index.faiss")
        )
        faiss.write_index(
            self.summary_index.index,
            os.path.join(self.storage_dir, "summary_index.faiss"),
        )
        faiss.write_index(
            self.text_index.index, os.path.join(self.storage_dir, "text_index.faiss")
        )
        # Save document texts
        with open(os.path.join(self.storage_dir, "title_texts.txt"), "w") as f:
            f.write("\n".join(self.title_documents))
        with open(os.path.join(self.storage_dir, "summary_texts.txt"), "w") as f:
            f.write("\n".join(self.summary_documents))
        with open(os.path.join(self.storage_dir, "chunk_texts.txt"), "w") as f:
            f.write("\n".join(self.text_documents))
        print("Chunks have been saved in FAISS indexes for title, summary, and text.")

    def load_chunks_separate_indexes(self):
        """
        Loads the FAISS indexes and documents from storage.
        """
        self.title_index = self.read_index("title_index.faiss")
        self.summary_index = self.read_index("summary_index.faiss")
        self.text_index = self.read_index("text_index.faiss")
        # print(self.text_index)

        with open(os.path.join(self.storage_dir, "title_texts.txt"), "r") as f:
            self.title_documents = f.read().splitlines()
        with open(os.path.join(self.storage_dir, "summary_texts.txt"), "r") as f:
            self.summary_documents = f.read().splitlines()
        with open(os.path.join(self.storage_dir, "chunk_texts.txt"), "r") as f:
            self.text_documents = f.read().splitlines()

        print("FAISS indexes and documents have been loaded from storage.")

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
        # query_embedding = self.model.encode(query)
        distances, indices = index.search(query, k)

        return [
            {"text": texts[idx], "distance": distances[0][i]}
            for i, idx in enumerate(indices[0])
        ]

    # def similarity_search(query, index, texts, k=5):
    #     distances, indices = index.search(query, k)
    #     results = [(texts[i], distances[0][j]) for j, i in enumerate(indices[0])]
    #     return results

    def bm25_search(self, query, texts, k=5):
        # Tokenize the texts and the query
        tokenized_texts = [doc.split() for doc in texts]
        bm25 = BM25Okapi(tokenized_texts)
        tokenized_query = query.split(" ")

        # BM25 search
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n_indices = bm25.get_top_n(tokenized_query, range(len(texts)), n=k)

        results = [
            {"text": texts[i], "distance": bm25_scores[i]} for i in top_n_indices
        ]
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
        filtered_results = [
            results for results in combined_results if results["distance"] > threshold
        ]
        # Sort results by score (assuming each result has a 'score' field)
        reranked_results = sorted(
            filtered_results, key=lambda x: x["distance"], reverse=True
        )
        result = self.remove_duplicates(
            reranked_results[:top_n]
            if len(reranked_results) > top_n
            else reranked_results
        )

        pprint(result)
        # Return the top N results or all results if fewer than N
        return result

    def retrieve_and_rerank(self, query, top_n=15):
        text_query = query
        query = self.model.encode([query])
        # Retrieve results from different indexes using each retriever
        # title_bm25_results = self.bm25_retrieve(query, self.title_index)
        # summary_embedding_results = self.embedding_retrieve(
        #     query, self.summary_index, self.model
        # )

        results = []

        faiss_results = self.faiss_retrieve(query, self.text_documents, self.text_index)
        results.extend(faiss_results)

        bm25_result = self.bm25_search(text_query, self.text_documents)
        results.extend(bm25_result)

        # Combine and rerank
        final_results = self.rerank_results(
            # title_bm25_results, summary_embedding_results,
            results,
            top_n,
        )

        return final_results


ai = KnowledgeBase()


# # Example usage:
# text = """Astronomy: The Milky Way galaxy is not just a collection of stars; it’s a vast, rotating system that includes planets, moons, asteroids, comets, and cosmic dust. It spans about 100,000 light-years in diameter and contains our solar system. Understanding the Milky Way helps astronomers learn about the structure and evolution of galaxies, the distribution of different types of stars, and the potential for extraterrestrial life. The center of the Milky Way harbors a supermassive black hole, known as Sagittarius A*, which plays a crucial role in the dynamics of our galaxy.

# Ancient Egypt: The Great Pyramid of Giza, also known as the Pyramid of Khufu, is a monumental feat of engineering and architecture. Built during the Fourth Dynasty for the Pharaoh Khufu, it stood as the tallest man-made structure in the world for over 3,800 years, until the construction of the Lincoln Cathedral in England in 1311 AD. The pyramid was originally covered in casing stones made of highly polished Tura limestone, which reflected the sun’s light and made the pyramid shine like a ‘gem.’ Inside, the pyramid contains a series of complex passageways and chambers, including the King’s Chamber, the Queen’s Chamber, and the enigmatic Grand Gallery.

# Technology: Quantum computers operate on the principles of quantum mechanics, using qubits instead of classical bits. These qubits can exist in multiple states simultaneously (superposition) and can be entangled, allowing quantum computers to process a vast number of possibilities at once. This capability enables them to solve certain problems much faster than classical computers, such as factoring large numbers, simulating molecular structures for drug discovery, and optimizing complex systems. However, building a practical quantum computer poses significant technical challenges, including error correction and maintaining qubit coherence.

# History: The first known democracy was established in Athens around 508-507 BC by the reforms of Cleisthenes. This Athenian democracy was a direct democracy, where citizens could participate in decision-making directly rather than through elected representatives. The main institutions of Athenian democracy included the Assembly (Ekklesia), the Council of 500 (Boule), and the People's Court (Heliaia). Citizens could speak, propose laws, and vote on important issues. While Athenian democracy was groundbreaking, it was also limited, as it excluded women, slaves, and non-citizens from participation.

# Art: Vincent van Gogh, one of the most influential figures in Western art, sold only one painting during his lifetime: "The Red Vineyard." Despite his struggles with mental health and poverty, van Gogh produced more than 2,000 artworks, including around 860 oil paintings, many of which were created in the last two years of his life. His expressive and emotive use of color, brushwork, and composition profoundly impacted modern art. Van Gogh's works, such as "Starry Night," "Sunflowers," and "The Bedroom," are celebrated for their intensity, emotion, and innovative approach to capturing the essence of the natural world.

# Marine Biology: Certain species of jellyfish, such as Turritopsis dohrnii, possess the ability to revert to an earlier stage of their life cycle through a process called transdifferentiation. When faced with environmental stress, injury, or old age, these jellyfish can transform their cells back into a polyp state, essentially restarting their life cycle. This process makes them biologically immortal, as they can theoretically repeat this cycle indefinitely. Studying these jellyfish provides insights into cellular regeneration, aging, and the potential for applications in medical science.

# Psychology: The placebo effect is a fascinating phenomenon in which a patient experiences a perceived or actual improvement in their condition after receiving a treatment with no therapeutic effect. This effect underscores the powerful influence of the mind on physical health. Placebos are often used in clinical trials to test the efficacy of new treatments by comparing them to an inert substance. Understanding the placebo effect helps researchers design better experiments and explore the psychological and physiological mechanisms that contribute to healing and well-being.

# Space Exploration: The Voyager 1 spacecraft, launched by NASA in 1977, has traveled farther from Earth than any other human-made object. It carries a golden record containing sounds and images representing life and culture on Earth, intended as a message to potential extraterrestrial civilizations. In 2012, Voyager 1 entered interstellar space, providing valuable data about the heliosphere and the transition to the interstellar medium. Its twin, Voyager 2, also continues to send back scientific information from its journey through space.

# Economics: The concept of supply and demand is a fundamental principle in economics that describes how the quantity of a good or service available (supply) and the desire for that good or service (demand) interact to determine its price. When demand for a product increases and supply remains constant, prices tend to rise. Conversely, if supply increases and demand remains constant, prices tend to fall. This dynamic is essential for understanding market behavior, price formation, and the allocation of resources in an economy.

# Literature: William Shakespeare’s works have been translated into over 80 languages, showcasing the universal appeal and enduring relevance of his plays and poetry. His writings explore timeless themes such as love, power, jealousy, betrayal, and the human condition. From the tragic depths of "Hamlet" and "Macbeth" to the comedic brilliance of "A Midsummer Night's Dream" and "Much Ado About Nothing," Shakespeare's mastery of language and storytelling continues to captivate audiences worldwide. His influence extends beyond literature to art, music, theater, and popular culture.

# Geography: The Amazon River, flowing through South America, is the largest river by discharge volume of water in the world. It discharges more water than the next seven largest rivers combined. The river basin is home to the Amazon rainforest, the largest tropical rainforest on Earth, which supports unparalleled biodiversity. The Amazon River spans approximately 6,000 kilometers (4,000 miles) and provides a critical waterway for transportation, commerce, and sustenance for millions of people living in the region.

# Music: Ludwig van Beethoven, one of the greatest composers in Western music history, continued to compose even after becoming completely deaf. Despite his hearing loss, he created some of his most iconic works, including the Ninth Symphony, which features the famous "Ode to Joy" choral finale. Beethoven's ability to innovate and push the boundaries of classical music left a profound legacy that influenced generations of composers and musicians. His compositions are celebrated for their emotional depth, structural complexity, and enduring appeal.

# Sports: The ancient Olympic Games, held in Olympia, Greece, from 776 BC to 393 AD, were a series of athletic competitions among representatives of various city-states. Events included running, long jump, shot put, javelin, boxing, and chariot racing. The games were held every four years, a period known as an Olympiad, and were part of a festival honoring Zeus. Victorious athletes were celebrated and often immortalized in statues and poetry. The modern Olympic Games, revived in 1896, continue this tradition of international athletic competition.

# Philosophy: Existentialism is a philosophical movement that emerged in the 19th and 20th centuries, focusing on individual freedom, choice, and responsibility. It explores the idea that humans create their own meaning and purpose in an indifferent or absurd universe. Key existentialist thinkers include Søren Kierkegaard, Friedrich Nietzsche, Jean-Paul Sartre, and Simone de Beauvoir. Existentialism emphasizes the importance of personal authenticity, self-awareness, and the courage to confront existential anxiety and the inevitability of death.

# Physics: The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space, time, and gravity. It consists of two parts: special relativity and general relativity. Special relativity introduced the concept that the laws of physics are the same for all observers in uniform motion and led to the famous equation E=mc². General relativity extended this to include gravity, describing it as the curvature of spacetime caused by mass and energy. These theories have been confirmed by numerous experiments and have profound implications for our understanding of the universe.

# Biology: DNA (deoxyribonucleic acid) is the molecule that carries genetic information in all living organisms. It consists of two strands that coil around each other to form a double helix. The sequence of nucleotides (adenine, thymine, cytosine, and guanine) in DNA determines the genetic instructions for the development, functioning, growth, and reproduction of organisms. The discovery of the structure of DNA by James Watson and Francis Crick, based on the work of Rosalind Franklin and others, was a pivotal moment in biology, leading to advancements in genetics, medicine, and biotechnology.

# Chemistry: The periodic table, created by Dmitri Mendeleev, organizes chemical elements based on their atomic number, electron configuration, and recurring chemical properties. Elements are arranged in rows (periods) and columns (groups) to highlight similarities in their properties. The periodic table serves as a fundamental tool for chemists, providing a framework for understanding chemical behavior, predicting reactions, and exploring new elements. It has undergone several revisions as new elements have been discovered and our understanding of atomic structure has evolved.

# Mathematics: The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, starting from 0 and 1. The sequence is named after Leonardo of Pisa, known as Fibonacci, who introduced it to the Western world in his 1202 book "Liber Abaci."""
# chunks = ai.semantic_chunking(text)


# pprint(chunks)


ai.retrieve_and_rerank("jellyfish")
