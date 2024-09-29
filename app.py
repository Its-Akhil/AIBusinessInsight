import streamlit as st
from PyPDF2 import PdfReader
import os
from rag_knowledge_base import RAGKnowledgeBase

st.set_page_config(layout="wide", page_title="AI-Driven Business Strategy Platform")


import pandas as pd
import numpy as np
import plotly.express as px
from faiss import IndexFlatL2
import websockets
import json
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

# Load environment variables
load_dotenv()


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

    def add_to_knowledge_base(self, file_path, category):
        self.kb.add_to_knowledge_base(file_path, category)

    def generate_with_rag(self, prompt, k=5):
        if not self.kb.documents:
            print("No documents in knowledge base. Generating response without RAG.")
            return self.generate_without_rag(prompt)

        selected_docs = st.session_state.get("selected_docs", [])
        print(f"Selected documents: {selected_docs}")

        results, distances = self.kb.query_knowledge_base(prompt, k=k)
        print(f"Query results: {results}")
        print(f"Query distances: {distances}")

        if not results:
            print(
                "No results found in the knowledge base. Generating response without RAG."
            )
            return self.generate_without_rag(prompt)

        if selected_docs:
            filtered_results = []
            filtered_distances = []
            for result, distance in zip(results, distances):
                if any(doc in result["path"] for doc in selected_docs):
                    filtered_results.append(result)
                    filtered_distances.append(distance)
            results = filtered_results
            distances = filtered_distances

        print(f"Filtered results: {results}")

        context = "Relevant information:\n"
        for doc, distance in zip(results, distances):
            relevance = 1 / (1 + distance) if distance != float("inf") else 0
            context += f"- {doc['filename']} (relevance: {relevance:.2f}):\n"
            try:
                with open(doc["path"], "r", encoding="utf-8") as f:
                    content = f.read()
                    context += f"  {content[:200]}...\n"
            except Exception as e:
                print(f"Error reading file {doc['path']}: {str(e)}")

        print(f"Generated context: {context}")

        if context == "Relevant information:\n":
            print("No relevant context found. Generating response without RAG.")
            return self.generate_without_rag(prompt)

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
        self.ws = await websockets.connect(self.url)

    async def receive_data(self):
        encrypted_message = await self.ws.recv()
        decrypted_message = decrypt_data(encrypted_message, username="StreamlitApp")
        if decrypted_message is None:
            print("Error decrypting message from WebSocket")
            return None
        try:
            return json.loads(decrypted_message)
        except json.JSONDecodeError:
            print("Received message is not valid JSON")
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
            "Customer Count",
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


def upload_pdf(kb):
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None and "last_uploaded_file" not in st.session_state:
        try:
            # Create a StringIO object to hold the extracted text
            output_string = StringIO()

            # Extract text from PDF using pdfminer.six
            extract_text_to_fp(
                uploaded_file,
                output_string,
                laparams=LAParams(),
                output_type="text",
                codec="utf-8",
            )

            # Get the extracted text
            text = output_string.getvalue()

            # Save PDF content to a temporary file using UTF-8 encoding
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "w", encoding="utf-8") as temp_file:
                temp_file.write(text)

            # Add to knowledge base
            kb.add_to_knowledge_base(temp_file_path, "uploaded_documents")

            # Remove temporary file
            os.remove(temp_file_path)

            st.success(
                f"File {uploaded_file.name} has been added to the knowledge base."
            )
            print(f"Added document: {uploaded_file.name}")
            print(f"Total documents after addition: {len(kb.documents)}")

            # Mark this file as processed
            st.session_state.last_uploaded_file = uploaded_file.name
        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {str(e)}")
            print(f"Error details: {e}")
    elif "last_uploaded_file" in st.session_state:
        st.info(
            f"File {st.session_state.last_uploaded_file} has already been processed."
        )


def main():
    # Initialize CustomAI and RAGKnowledgeBase
    ai = CustomAI()
    kb = ai.kb

    # Initialize data in session state if it doesn't exist
    if "data" not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=["timestamp", "value"])

    # Left sidebar
    with st.sidebar:
        st.image("assets/images/icon.png", width=200)
        st.title("AI Business Insight")
        colored_header(
            "Business Query",
            description="Ask me anything about your business",
            color_name="blue-70",
        )
        user_query = st.text_input("Enter your business question:")
        if st.button("Get Insights", key="query_button"):
            with st.spinner("Analyzing..."):
                response = process_query(user_query)
            st.success("Analysis complete!")
            st.write("AI Response:", response)
            # Store the query and response in session state
            st.session_state.last_query = user_query
            st.session_state.last_response = response

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

    # Toggle for right sidebar
    if "show_right_sidebar" not in st.session_state:
        st.session_state.show_right_sidebar = False

    if st.button("Toggle Knowledge Base Management"):
        st.session_state.show_right_sidebar = not st.session_state.show_right_sidebar

    if st.session_state.show_right_sidebar:
        # Right sidebar content
        st.sidebar.title("Knowledge Base Management")

        # Upload PDF
        st.sidebar.subheader("Upload PDF")
        upload_pdf(kb)

        # List and select documents
        st.sidebar.subheader("Available Documents")
        documents = kb.list_documents()
        print(f"Available documents: {documents}")

        selected_docs = []
        for doc in documents:
            if st.sidebar.checkbox(doc, value=True):
                selected_docs.append(doc)
        print(f"Selected documents: {selected_docs}")

        # Store selected documents in session state
        st.session_state.selected_docs = selected_docs

    ws_client = WebSocketClient(os.getenv("WEBSOCKET_URL"))
    data_processor = DataProcessor()

    async def update_dashboard():
        await ws_client.connect()

        while True:
            new_data = await ws_client.receive_data()
            if new_data is None:
                await asyncio.sleep(2)
                continue

            new_row = pd.DataFrame([new_data])
            st.session_state.data = pd.concat(
                [st.session_state.data, new_row], ignore_index=True
            ).tail(100)
            processed_data = data_processor.process_data(st.session_state.data)

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

            # Customize metric cards
            style_metric_cards(
                background_color="transparent",
                border_left_color="#686664",
                border_color="#686664",
                box_shadow=None,
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
                    st.write("Used Documents:")
                    for doc in st.session_state.get("selected_docs", []):
                        st.write(f"- {doc}")

            await asyncio.sleep(2)

    # Run the dashboard update in a separate thread
    threading.Thread(
        target=lambda: asyncio.run(update_dashboard()), daemon=True
    ).start()

    # Display the current state of the dashboard
    if not st.session_state.data.empty:
        processed_data = data_processor.process_data(st.session_state.data)

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

        # Customize metric cards
        style_metric_cards(
            background_color="transparent",
            border_left_color="#686664",
            border_color="#686664",
            box_shadow=None,
        )

        # Create and update chart
        fig = create_dashboard_chart(processed_data)
        chart_placeholder.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
