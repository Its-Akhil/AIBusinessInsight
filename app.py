import streamlit as st
from PyPDF2 import PdfReader
import os
from ragWithNER import (
    KnowledgeBase,
    SentenceTransformerWrapper,
)
from pprint import pprint
from LstmModelCreator import LSTMModel


st.set_page_config(layout="wide", page_title="AI-Driven Business Strategy Platform")
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

import time
import pandas as pd
import numpy as np
import plotly.express as px
from faiss import IndexFlatL2
import websockets
import json, logging
import asyncio
from encryption_utils import decrypt_data, encrypt_data
import os
from together import Together
from dotenv import load_dotenv
import plotly.subplots as sp
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import base64
import io
from io import StringIO
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import threading
from pathlib import Path
from queue import Queue

# Load environment variables
load_dotenv()

# Global variables
global_data = pd.DataFrame(columns=["timestamp", "value"])
data_lock = threading.Lock()
update_queue = Queue()

# Global counter for data storage iterations
data_storage_counter = 0


def read_encryption_key():
    key_file_path = os.path.join(os.path.dirname(__file__), "encryption_key.key")
    with open(key_file_path, "rb") as key_file:
        return key_file.read().strip()


# Get the encryption key from the file
ENCRYPTION_KEY = read_encryption_key()
if not ENCRYPTION_KEY:
    raise ValueError("Encryption key file is empty or not found")


# Ensure the key is in the correct format for Fernet
def ensure_fernet_key(key):
    if len(key) == 32:
        return base64.urlsafe_b64encode(key)
    elif len(key) == 44 and key.endswith(b"="):
        return key
    else:
        raise ValueError(
            "Encryption key must be 32 bytes or 44 characters ending with '='"
        )


ENCRYPTION_KEY = ensure_fernet_key(ENCRYPTION_KEY)


# Initialize Together AI LLM
from rag_knowledge_base import RAGKnowledgeBase


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


# qp = QueryProcessor()


class SentenceTransformerWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts)


class CustomAI:
    def __init__(self):
        api_key = os.getenv("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        self.client = Together(api_key=api_key)

        #
        self.kb = KnowledgeBase()  # Change this line to use KnowledgeBase

        #
        print(f"Loaded {len(self.kb.documents)} documents into the knowledge base.")

        self.last_kb_update = time.time()

    def add_to_knowledge_base(self, file_path, fname):
        self.kb.add_to_knowledge_base(file_path, fname)

    def check_and_update_kb(self):
        current_time = time.time()
        if current_time - self.last_kb_update > 300:  # Update every 5 minutes
            # print("Updating knowledge base with new encrypted data...")
            # self.kb.add_encrypted_data_to_knowledge_base()
            self.kb.load_knowledge_base()
            self.last_kb_update = current_time

    def generate_with_rag(self, prompt, k=10):
        self.check_and_update_kb()
        if not self.kb.documents:
            print("No documents in knowledge base. Generating response without RAG.")
            return self.generate_without_rag(prompt)

        selected_docs = st.session_state.get("selected_docs", [])
        print(f"Selected documents: {selected_docs}")

        context = self.kb.retrieve_and_rerank(prompt, k)

        if not context:
            print(
                "No results found in the knowledge base. Generating response without RAG."
            )
            return self.generate_without_rag(prompt)

        # pprint(context)

        augmented_prompt = f"{context}\n\nUser query: {prompt}\n\nBased on the above information, please respond to the user query:"

        return self.generate_without_rag(augmented_prompt)

    def generate_without_rag(self, prompt):

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

    def generate(self, prompt):
        return self.generate_without_rag(prompt)


ai = CustomAI()

# Initialize FAISS index
index = IndexFlatL2(256)  # Adjust dimension as needed


class WebSocketClient:
    def __init__(self, url):
        self.url = url
        self.ws = None

    async def connect(self):
        try:
            self.ws = await websockets.connect(self.url)
            # logging.info("WebSocket connection established")
        except Exception as e:
            # logging.error(f"WebSocket connection failed: {e}")
            self.ws = None

    async def receive_data(self):
        if self.ws is None:
            return None
        try:
            encrypted_message = await self.ws.recv()
            # logging.debug(f"Received encrypted message: {encrypted_message[:50]}...")
            decrypted_message = decrypt_data(encrypted_message, username="StreamlitApp")
            if decrypted_message is None:
                # logging.error("Error decrypting message from WebSocket.")
                # logging.error(f"Full encrypted message: {encrypted_message}")
                return None
            else:
                # logging.debug(
                # f"Successfully decrypted message: {decrypted_message[:50]}..."
                # )
                return json.loads(decrypted_message)
        except websockets.exceptions.ConnectionClosed:
            # logging.warning("WebSocket connection closed")
            self.ws = None
            return None
        except json.JSONDecodeError as e:
            # logging.error(f"Error parsing JSON from decrypted message: {e}")
            # logging.error(f"Decrypted message: {decrypted_message}")
            return None
        except Exception as e:
            # logging.error(f"Unexpected error receiving data: {e}")
            return None


class DataProcessor:
    @staticmethod
    def process_data(data):
        data["MA7"] = data["sales"].rolling(window=7).mean()
        data["MA30"] = data["sales"].rolling(window=30).mean()

        # Check if we have enough data to predict
        if len(previous_20_rows) >= 20:
            # print(
            #     len(previous_20_rows), len(previous_20_rows[-20:]), "\n\n\n\n\n\n\n\n\n"
            # )

            try:
                # input_data = np.array([[row['input_value'], row['customers'], row['average_order_value'], row['customer_satisfaction']]])

                dictList = []

                # # Collect the last 10 entries from the data buffer
                # sales = [buffer["sales"] for buffer in previous_20_rows[-10:]]
                # customers = [buffer["customers"] for buffer in previous_20_rows[-10:]]
                # average_order_value = [
                #     buffer["average_order_value"] for buffer in previous_20_rows[-10:]
                # ]
                # customer_satisfaction = [
                #     buffer["customer_satisfaction"] for buffer in previous_20_rows[-10:]
                # ]

                # # Create a 2D array where each row is a feature
                # input_data = np.array(
                #     [sales, customers, average_order_value, customer_satisfaction]
                # ).T

                # # Scale the data
                # scaled_data = lstm_model.scaler.fit_transform(input_data)

                # # Optionally, you can unpack the scaled data back into separate variables
                # (
                #     scaled_sales,
                #     scaled_customers,
                #     scaled_average_order_value,
                #     scaled_customer_satisfaction,
                # ) = (
                #     scaled_data[:, 0],  # Sales
                #     scaled_data[:, 1],  # Customers
                #     scaled_data[:, 2],  # Average Order Value
                #     scaled_data[:, 3],  # Customer Satisfaction
                # )

                # input_data_lstm = []
                # for i in range(len(scaled_data) - lstm_model.time_steps):
                #     input_data_lstm.append(scaled_data[i : i + lstm_model.time_steps])

                # input_data_lstm = np.array(input_data_lstm)

                # predicted_value = lstm_model.predict(input_data_lstm)

                # print("Successful prediction:", predicted_value)
                # anomaly = np.abs(row["actual_value"] - predicted_value) > threshold

                # Store index of anomalies
                # if anomaly:
                # anomaly_indices.append(len(bar_data) - 1)
                # "sales": [i["sales"] for i in previous_20_rows],
                # "customers": [i["customers"] for i in previous_20_rows],
                # "average_order_value": [
                #     i["average_order_value"] for i in previous_20_rows
                # ],
                # "customer_satisfaction": [
                #     i["customer_satisfaction"] for i in previous_20_rows
                # ],
                # x, y = lstm_model.preprocess_data(dictList)
                dictList = [
                    {key: value for key, value in i.items() if key != "timestamp"}
                    for i in previous_20_rows
                ]
                predicted_sales = predict_from_buffer(lstm_model, dictList)
                print(
                    "\n\n\n\n\n\n<<<<Predicted Sales:>>>>",
                    len(predicted_sales.tolist()),
                    type(data),
                )
                previous_20_predictions.extend(predicted_sales.tolist())

                n = len(previous_20_predictions)
                data["predicted_sales"] = data["sales"].copy()

                if n <= len(data):
                    data["predicted_sales"].iloc[-n:] = previous_20_predictions[-n:]
                else:
                    try:
                        data["predicted_sales"].iloc[-30:] = previous_20_predictions[
                            -30:
                        ]
                    except Exception as e:
                        print(f"Exception <><><> : {e}")
                    print("Not enough rows in 'data' to replace with predictions.")

                # dictList = np.tile(dictList, (1, 7))
                # .reshape(1, 1, 7)

                # # dicList = np.array(

                # #     [
                # #         [value for key, value in i.items() if key != "timestamp"]
                # #         for i in previous_20_rows
                # #     ]
                # # )
                # # dictList = np.array(dictList)
                # # print(dictList)
                # data["predicted_sales"] = lstm_model.predict(
                #     dictList,
                # )
            except Exception as e:
                print(f"\n\n\n\n\n\n\nPrediction error \n\n\n\n\n\n : {e}")
        if "predicted_sales" not in data:
            data["predicted_sales"] = data["sales"]
        return data

        # if len(data_buffer) >= rows_to_retrain:
        #     # Retrain the model using the received data
        #     lstm_model.retrain_lstm("mock_data.txt", data_buffer)
        #     data_buffer.clear()  # Clear the buffer after retraining

    # Update previous rows for prediction
    # if len(previous_20_rows) >= 20:
    #         previous_20_rows.pop(0)
    #         if len(previous_20_predictions) >= 20:
    #             previous_20_predictions.pop(0)

    #     previous_20_rows.append(row)

    #     # Check if we have enough data to predict
    #     if len(data_buffer) >= lstm_model.time_steps:
    #         try:
    #             predicted_sales = predict_from_buffer(
    #                 lstm_model,
    #                 [
    #                     {
    #                         key: value
    #                         for key, value in i.items()
    #                         if key != "timestamp"
    #                     }
    #                     for i in data_buffer
    #                 ],
    #             )
    #             print("Predicted Sales:", predicted_sales)
    #         except Exception as e:
    #             print(f"Prediction error: {e}")

    #     if len(data_buffer) >= rows_to_retrain:
    #         # Retrain the model using the received data
    #         lstm_model.retrain_lstm("mock_data.txt", data_buffer)
    #         data_buffer.clear()  # Clear the buffer after retraining


def upload_document(kb):
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file is not None and "last_uploaded_file" not in st.session_state:
        try:
            if uploaded_file.type == "application/pdf":
                # PDF processing
                output_string = StringIO()
                extract_text_to_fp(
                    uploaded_file,
                    output_string,
                    laparams=LAParams(),
                    output_type="text",
                    codec="utf-8",
                )
                text = output_string.getvalue()
            else:
                # TXT processing
                text = uploaded_file.getvalue().decode("utf-8")

            kb.add_to_knowledge_base(text, uploaded_file.name)

            st.success(
                f"File {uploaded_file.name} has been added to the knowledge base."
            )

            # Mark this file as processed
            st.session_state.last_uploaded_file = uploaded_file.name
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            print(f"Error details: {e}")
    elif "last_uploaded_file" in st.session_state:
        st.info(
            f"File {st.session_state.last_uploaded_file} has already been processed."
        )


# Add custom CSS
st.markdown(
    """
<style>
    .stApp {
        max-width: 100%;
        padding: 1rem;
        margin: 0 auto;
    }
    .stMetricCard {
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)


def create_dashboard_chart(data):
    fig = sp.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Sales Trend",
            "Customer Count",
            "Average Order Value",
            "Customer Satisfaction",
        ),
        vertical_spacing=0.21,
    )
    fig.add_trace(
        go.Scatter(
            x=data["timestamp"], y=data["predicted_sales"], name="Predicted Sales"
        ),
        row=1,
        col=1,
    )
    # Add traces for each subplot
    fig.add_trace(
        go.Scatter(x=data["timestamp"], y=data["sales"], name="Sales"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data["timestamp"], y=data["MA7"], name="7-day MA"), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=data["timestamp"], y=data["MA30"], name="30-day MA"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data["timestamp"], y=data["customers"], name="Customers"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=data["timestamp"], y=data["average_order_value"], name="AOV"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=data["timestamp"], y=data["customer_satisfaction"], name="CSAT"),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=900,
        title_text="Business Metrics Dashboard",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgrey")
    # print("\n\n\n\n created Fig \n\n\n\n\n")
    return fig


from encryption_utils import encrypt_data, decrypt_data


def process_query(query):
    # Encrypt the query before sending it to the LLM
    encrypted_query = encrypt_data(query, username="StreamlitApp")

    # Decrypt the query before processing
    decrypted_query = decrypt_data(encrypted_query, username="StreamlitApp")

    # Process the decrypted query using RAG
    response = ai.generate_with_rag(decrypted_query)
    # Encrypt the response before returning
    encrypted_response = encrypt_data(response, username="StreamlitApp")

    # Decrypt the response before displaying
    return decrypt_data(encrypted_response, username="StreamlitApp")


def store_encrypted_data(data, case_name):
    global data_storage_counter
    base_dir = Path("encrypted_data")
    case_dir = base_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    file_path = case_dir / f"{case_name}_data.json"

    try:
        if file_path.exists():
            with open(file_path, "r") as f:
                encrypted_content = f.read()

            decrypted_content = decrypt_data(encrypted_content, username="StreamlitApp")
            if decrypted_content is None:
                print(f"Decryption failed for {file_path}. Starting with empty data.")
                existing_data = []
            else:
                try:
                    existing_data = json.loads(decrypted_content)
                except json.JSONDecodeError:
                    print(
                        f"Failed to parse JSON from {file_path}. Starting with empty data."
                    )
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(data)

        # Increment the counter
        data_storage_counter += 1

        # Save data every 20th iteration or if it's the first iteration
        if data_storage_counter % 20 == 0 or data_storage_counter == 1:
            json_data = json.dumps(existing_data)
            encrypted_data = encrypt_data(json_data, username="StreamlitApp")

            with open(file_path, "w") as f:
                f.write(encrypted_data)

            print(
                f"Data stored successfully in {file_path} (Iteration: {data_storage_counter})"
            )
        else:
            print(f"Data added to memory buffer (Iteration: {data_storage_counter})")

    except Exception as e:
        print(f"Error handling data: {str(e)}")

    # If we've reached 20 iterations, reset the counter
    if data_storage_counter == 20:
        data_storage_counter = 0


async def fetch_data(ws_client):
    while True:
        if ws_client.ws is None:
            print("Attempting to establish WebSocket connection...")
            await ws_client.connect()
            if ws_client.ws is None:
                print("Failed to connect. Retrying in 5 seconds...")
                await asyncio.sleep(5)
                continue

        new_data = await ws_client.receive_data()
        # print("\n\n\n\n\n\n", new_data, "\n\n\n\n\n\n\n")
        if new_data is None:
            await asyncio.sleep(2)
            continue

        # if len(previous_20_rows) >= 40:
        #     previous_20_rows.pop(0)

        previous_20_rows.append(new_data)

        store_encrypted_data(new_data, "realtime_data")
        # print(f"New data received: {new_data}")
        update_queue.put(new_data)
        await asyncio.sleep(2)


def process_data():
    global global_data
    while True:
        if not update_queue.empty():
            new_data = update_queue.get()
            # print(f"Processing new data: {new_data}")  # Debugging line
            new_row = pd.DataFrame([new_data])
            with data_lock:
                global_data = pd.concat([global_data, new_row], ignore_index=True).tail(
                    100
                )
                # print(f"Updated global_data: {global_data}")  # Debugging line
        time.sleep(0.1)


lstm_model = LSTMModel()
lstm_model.load_model("lstm_model")


# Initialize the plot data
bar_data = []
predicted_values = []
anomaly_indices = []  # List to store indices of anomalies

# Buffer for storing received data (for retraining every 20 rows)
data_buffer = []
rows_to_retrain = 50
threshold = 0.5

previous_20_rows = []
previous_20_predictions = []


# Load and preprocess dataset
def load_dataset():
    df = pd.read_csv(
        "mock_data.csv",
        sep=",",  # Use comma as the separator
        header=None,  # No header in your data
        names=[
            "dt",
            "sales",
            "customers",
            "average_order_value",
            "customer_satisfaction",
        ],  # Define column names
        parse_dates=["dt"],  # Parse the 'dt' column directly
        low_memory=False,
        na_values=["nan", "?"],  # Handle NaN values
    )
    df = df.iloc[1:]

    df["dt"] = pd.to_datetime(df["dt"])
    df.set_index("dt", inplace=True)
    # Check if the index is a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "The index is not a DatetimeIndex. Please check the 'dt' column."
        )

    for j in range(df.shape[1]):  # Loop through all columns
        df.iloc[:, j] = pd.to_numeric(
            df.iloc[:, j], errors="coerce"
        )  # Convert to numeric, set non-convertibles to NaN

    # Fill NaN values with the mean of each column
    for j in range(df.shape[1]):  # Fill NaN values for numeric columns
        df.iloc[:, j] = df.iloc[:, j].fillna(df.iloc[:, j].mean())

    # Resample the data hourly and take the mean
    df_resample = df.resample("H").mean()  # Use 'H' for hourly resampling
    return df_resample


# def predict(self, input_data):

#     if len(data_buffer) < self.time_steps:
#         raise ValueError("Not enough data in the buffer for prediction.")
#     try:
#         # input_data_scaled = self.scaler.transform(input_data)
#         input_data_reshaped = input_data.reshape(
#             (1, self.time_steps, input_data.shape[1])
#         )
#         prediction = self.model.predict(input_data_reshaped)
#         print(prediction, "\n\n\n\n\n\n\n\n\n")
#         return self.scaler.inverse_transform([[prediction[0][0]]])[0][0]
#     except Exception as e:
#         raise ValueError(f"Error during prediction ><><>\n\n\n\n\n: {e}")


# Format row to JSON
def format_row_to_json(row):
    return ", ".join([f"{key}={value}" for key, value in row.items()])


# Handle anomaly detection
def Anomaly_detected(previous_20_rows, predicted_20_values):
    previous_rows_json = [format_row_to_json(row) for row in previous_20_rows]
    predicted_values_json = json.dumps(predicted_20_values)
    print(
        f"Anomaly detected!\nPrevious 20 rows: {previous_rows_json}\nPredicted values: {predicted_values_json}"
    )


# Load the dataset
df_resample = load_dataset()
scaled_data, scaler = lstm_model.preprocess_data(df_resample)


def predict_from_buffer(lstm_model, data_buffer):
    if len(data_buffer) < lstm_model.time_steps:
        raise ValueError("Not enough data in the buffer for prediction.")

    # Convert the data_buffer to a DataFrame for easy processing
    df = pd.DataFrame(data_buffer)
    feature_cols = [
        "sales",
        "customers",
        "average_order_value",
        "customer_satisfaction",
    ]

    # Ensure all necessary features are present
    if not all(col in df.columns for col in feature_cols):
        raise ValueError(
            f"DataBuffer must contain the following columns: {feature_cols}"
        )

    # Prepare the data for prediction
    data = df[feature_cols].values
    scaled_data = lstm_model.scaler.transform(
        data
    )  # Use the scaler from the trained model

    # Create input sequences for LSTM
    input_data = []
    for i in range(len(scaled_data) - lstm_model.time_steps):
        input_data.append(scaled_data[i : i + lstm_model.time_steps])

    input_data_lstm = np.array(input_data)

    # Check input shape
    print("Input data shape for prediction:", input_data_lstm.shape)

    # Perform prediction
    predictions = lstm_model.model.predict(input_data_lstm)

    # Inverse scale the predictions to get the original scale of sales
    predicted_sales = lstm_model.scaler.inverse_transform(
        np.concatenate([predictions, np.zeros((predictions.shape[0], 3))], axis=1)
    )[
        :, 0
    ]  # Take only the predicted sales

    return predicted_sales


def update_chart():
    chart_placeholder = st.empty()  # Create a placeholder for the chart

    # Create the bar chart
    bar_chart = go.Figure()

    # Add bars (actual values)
    bar_chart.add_trace(
        go.Bar(
            x=list(range(len(bar_data))),
            y=bar_data,
            marker=dict(color="blue"),
            name="Actual Data",
        )
    )

    # Add line (LSTM predicted values)
    bar_chart.add_trace(
        go.Scatter(
            x=list(range(len(predicted_values))),
            y=predicted_values,
            mode="lines+markers",
            name="LSTM Prediction",
            line=dict(color="green"),
        )
    )

    # Add markers for anomalies
    if anomaly_indices:
        bar_chart.add_trace(
            go.Scatter(
                x=anomaly_indices,
                y=[bar_data[i] for i in anomaly_indices],
                mode="markers",
                marker=dict(color="red", size=10),
                name="Anomalies",
            )
        )

    # Update the chart in the Streamlit placeholder
    chart_placeholder.plotly_chart(bar_chart, use_container_width=True)


async def receive_data():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        while True:
            # Receive data from WebSocket
            data = await websocket.recv()
            row = json.loads(decrypt_data(data))
            # print(f"Received row: {row}")  # Debug log
            bar_data.append(row)

            # Append received row to buffer
            data_buffer.append(row)

            # Update previous rows for prediction
            if len(previous_20_rows) >= 20:
                previous_20_rows.pop(0)

            previous_20_rows.append(row)

            # Check if we have enough data to predict
            if len(data_buffer) >= lstm_model.time_steps:
                try:
                    predicted_sales = lstm_model.predict(
                        # lstm_model,
                        [
                            {
                                key: value
                                for key, value in i.items()
                                if key != "timestamp"
                            }
                            for i in data_buffer
                        ],
                    )
                    print(
                        "_____________________________Predicted Sales:", predicted_sales
                    )
                except Exception as e:
                    print(f"Prediction error :> {e}")

            # if len(data_buffer) >= rows_to_retrain:
            #     # Retrain the model using the received data
            #     lstm_model.retrain_lstm("mock_data.txt", data_buffer)
            #     data_buffer.clear()  # Clear the buffer after retraining


# Background thread to run asyncio loop for WebSocket
def start_websocket_loop():
    asyncio.run(receive_data())


# Main Streamlit application
def main():
    global global_data, data_lock

    # Initialize CustomAI and RAGKnowledgeBase
    ai = CustomAI()
    kb = ai.kb

    # Start the data processing thread
    threading.Thread(target=process_data, daemon=True).start()

    # Left sidebar
    with st.sidebar:

        # Query section
        colored_header(
            "Business Query",
            description="Ask me anything about your business",
            color_name="blue-70",
        )
        user_query = st.text_input(
            "Enter your business question:", key="user_query_input"
        )
        if st.button("Get Insights", key="query_button"):
            with st.spinner("Analyzing..."):
                response = ai.generate_with_rag(user_query)
            st.success("Analysis complete!")
            st.write("AI Response:", response)
            st.session_state.last_query = user_query
            st.session_state.last_response = response

        # Advanced options (expandable)
        with st.expander("Advanced Options"):
            st.subheader("Document Selection")
            available_docs = kb.list_documents()
            selected_docs = st.multiselect(
                "Select documents to include in analysis:", available_docs
            )
            st.session_state.selected_docs = selected_docs

    # Main content
    colored_header(
        "Business Metrics Dashboard",
        description="Real-time insights",
        color_name="blue-70",
    )
    add_vertical_space(2)

    # Create placeholders for the chart and KPIs
    kpi_placeholder = st.empty()
    chart_placeholder = st.empty()
    rag_placeholder = st.empty()

    ws_client = WebSocketClient(os.getenv("WEBSOCKET_URL"))
    threading.Thread(
        target=lambda: asyncio.run(fetch_data(ws_client)), daemon=True
    ).start()

    # Main loop for updating the UI
    while True:
        with data_lock:
            current_data = global_data.copy()

        if not current_data.empty:

            processed_data = DataProcessor.process_data(current_data)

            with kpi_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Sales", f"${processed_data['sales'].sum():,.0f}")
                col2.metric(
                    "Avg Daily Customers",
                    f"{processed_data['customers'].mean():,.0f}",
                )
                col3.metric(
                    "Avg Order Value",
                    f"${processed_data['average_order_value'].mean():,.2f}",
                )
                col4.metric(
                    "Avg Satisfaction",
                    f"{processed_data['customer_satisfaction'].mean():.2f}/5.0",
                )

                fig = create_dashboard_chart(processed_data)
                st.plotly_chart(fig, use_container_width=True)

        if "last_query" in st.session_state and "last_response" in st.session_state:
            with rag_placeholder.container():
                st.subheader("Latest AI Insight")
                st.write(f"Query: {st.session_state.last_query}")
                st.write(f"Response: {st.session_state.last_response}")
                if st.session_state.get("selected_docs"):
                    st.write("Used Documents:")
                    for doc in st.session_state.selected_docs:
                        st.write(f"- {doc}")
        time.sleep(2)


if __name__ == "__main__":
    main()
