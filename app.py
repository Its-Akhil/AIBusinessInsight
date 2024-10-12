import streamlit as st
from PyPDF2 import PdfReader
import os
from rag_knowledge_base import RAGKnowledgeBase

st.set_page_config(layout="wide", page_title="AI-Driven Business Strategy Platform")

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


class CustomAI:
    def __init__(self):
        api_key = os.getenv("TOGETHER_API_KEY")
        if api_key is None:
            raise ValueError("TOGETHER_API_KEY environment variable is not set")
        self.client = Together(api_key=api_key)
        self.kb = RAGKnowledgeBase()
        print(f"Loaded {len(self.kb.documents)} documents into the knowledge base.")
        print(
            f"FAISS index status: {'Initialized' if self.kb.index is not None else 'Not initialized'}"
        )
        self.last_kb_update = time.time()

    def add_to_knowledge_base(self, file_path, category):
        self.kb.add_to_knowledge_base(file_path, category)

    def check_and_update_kb(self):
        current_time = time.time()
        if current_time - self.last_kb_update > 300:  # Update every 5 minutes
            print("Updating knowledge base with new encrypted data...")
            self.kb.add_encrypted_data_to_knowledge_base()
            self.kb.create_faiss_index()
            self.last_kb_update = current_time

    def generate_with_rag(self, prompt, k=10):
        self.check_and_update_kb()
        if not self.kb.documents:
            print("No documents in knowledge base. Generating response without RAG.")
            return self.generate_without_rag(prompt)

        selected_docs = st.session_state.get("selected_docs", [])
        print(f"Selected documents: {selected_docs}")

        results, indices = self.kb.query_knowledge_base(prompt, k=k)
        print(f"Query results: {results}")
        print(f"Query indices: {indices}")

        if not results:
            print(
                "No results found in the knowledge base. Generating response without RAG."
            )
            return self.generate_without_rag(prompt)

        if selected_docs:
            filtered_results = []
            for result in results:
                if any(doc in result.get("document", "") for doc in selected_docs):
                    filtered_results.append(result)

            if filtered_results:
                results = filtered_results
            else:
                print("No results after filtering. Using all results instead.")
                # Keep original results if filtering removes everything

        print(f"Filtered results: {results}")

        context = "Relevant information:\n"
        for i, doc in enumerate(results[:5], 1):  # Limit to top 5 results
            relevance = (
                1 / (1 + doc["semantic_distance"])
                if doc["semantic_distance"] != "N/A"
                else 0
            )
            context += f"- Document {i} (relevance: {relevance:.2f}):\n"
            content = doc.get("text", "")
            context += f"  {content[:200]}...\n"

        print(f"Generated context: {context}")

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
            logging.info("WebSocket connection established")
        except Exception as e:
            logging.error(f"WebSocket connection failed: {e}")
            self.ws = None

    async def receive_data(self):
        if self.ws is None:
            return None
        try:
            encrypted_message = await self.ws.recv()
            logging.debug(f"Received encrypted message: {encrypted_message[:50]}...")
            decrypted_message = decrypt_data(encrypted_message, username="StreamlitApp")
            if decrypted_message is None:
                logging.error("Error decrypting message from WebSocket.")
                logging.error(f"Full encrypted message: {encrypted_message}")
                return None
            else:
                logging.debug(
                    f"Successfully decrypted message: {decrypted_message[:50]}..."
                )
                return json.loads(decrypted_message)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("WebSocket connection closed")
            self.ws = None
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON from decrypted message: {e}")
            logging.error(f"Decrypted message: {decrypted_message}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error receiving data: {e}")
            return None


class DataProcessor:
    @staticmethod
    def process_data(data):
        data["MA7"] = data["sales"].rolling(window=7).mean()
        data["MA30"] = data["sales"].rolling(window=30).mean()
        return data


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
            "Customer `Coun`t",
            "Average Order Value",
            "Customer Satisfaction",
        ),
        vertical_spacing=0.21,
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

            # Print debug information
            print(f"Extracted text from {uploaded_file.name}:")
            print(text[:500])  # Print first 500 characters of the extracted text

            # Save content to a temporary file using UTF-8 encoding
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                temp_file.write(text)

            # Add to knowledge base
            kb.add_to_knowledge_base(temp_file_path, "uploaded_documents")

            # Print debug information
            print(f"Added document to knowledge base: {temp_file_path}")
            print(f"Total documents in knowledge base: {len(kb.documents)}")
            print(f"Documents in knowledge base: {kb.documents}")

            # Remove temporary file
            os.remove(temp_file_path)

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
        if new_data is None:
            await asyncio.sleep(2)
            continue

        # Store the encrypted data
        store_encrypted_data(new_data, "realtime_data")

        update_queue.put(new_data)
        await asyncio.sleep(2)


def process_data():
    global global_data
    while True:
        if not update_queue.empty():
            new_data = update_queue.get()
            new_row = pd.DataFrame([new_data])
            with data_lock:
                global_data = pd.concat([global_data, new_row], ignore_index=True).tail(
                    100
                )
        time.sleep(0.1)


def main():
    global global_data, data_lock

    # Initialize CustomAI and RAGKnowledgeBase
    ai = CustomAI()
    kb = ai.kb

    # Left sidebar
    with st.sidebar:
        st.image("assets/images/icon.png", width=200)
        st.title("AI Business Insight")

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
            # Store the query and response in session state
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

    # Add a section for RAG results and AI response
    rag_placeholder = st.empty()

    # Start background tasks
    ws_client = WebSocketClient(os.getenv("WEBSOCKET_URL"))
    threading.Thread(
        target=lambda: asyncio.run(fetch_data(ws_client)), daemon=True
    ).start()
    threading.Thread(target=process_data, daemon=True).start()

    # Main loop for updating the UI
    while True:
        with data_lock:
            current_data = global_data.copy()

        if not current_data.empty:
            processed_data = DataProcessor.process_data(current_data)

            # Update KPIs
            with kpi_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Sales", f"${processed_data['sales'].sum():,.0f}")
                col2.metric(
                    "Avg Daily Customers", f"{processed_data['customers'].mean():,.0f}"
                )
                col3.metric(
                    "Avg Order Value",
                    f"${processed_data['average_order_value'].mean():,.2f}",
                )
                col4.metric(
                    "Avg Satisfaction",
                    f"{processed_data['customer_satisfaction'].mean():.2f}/5.0",
                )

            # Create and update chart
            fig = create_dashboard_chart(processed_data)
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Update RAG results and AI response if available
        if "last_query" in st.session_state and "last_response" in st.session_state:
            with rag_placeholder.container():
                st.subheader("Latest AI Insight")
                st.write(f"Query: {st.session_state.last_query}")
                st.write(f"Response: {st.session_state.last_response}")
                if st.session_state.get("selected_docs"):
                    st.write("Used Documents:")
                    for doc in st.session_state.selected_docs:
                        st.write(f"- {doc}")

        time.sleep(2)  # Update every 2 seconds


if __name__ == "__main__":
    main()
