import asyncio
import websockets
import json
import random
import datetime
from encryption_utils import encrypt_data, decrypt_data
from dotenv import load_dotenv
import os, math


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


# Initialize variables to maintain the trend
sales_base = 3000
customers_base = 120
average_order_value_base = 100
customer_satisfaction_base = 4.3
time_step = 0  # To simulate time progression


async def send_mock_data(websocket, path):
    global time_step

    while True:
        # Introduce a sine wave trend with some noise for sales and customers
        sales_trend = (
            sales_base + 500 * math.sin(time_step / 10) + random.randint(-100, 100)
        )
        customers_trend = (
            customers_base + 20 * math.sin(time_step / 15) + random.randint(-10, 10)
        )

        # Linear trend with noise for average order value and customer satisfaction
        average_order_value_trend = (
            average_order_value_base + (time_step * 0.05) + random.uniform(-10, 10)
        )
        customer_satisfaction_trend = (
            customer_satisfaction_base + (time_step * 0.01) + random.uniform(-0.1, 0.1)
        )

        # Ensure values stay within realistic bounds
        sales_trend = max(1000, min(sales_trend, 5000))
        customers_trend = max(50, min(customers_trend, 200))
        average_order_value_trend = max(50, min(average_order_value_trend, 200))
        customer_satisfaction_trend = max(3.5, min(customer_satisfaction_trend, 5.0))

        # Generate mock business metrics with trends
        mock_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sales": int(sales_trend),
            "customers": int(customers_trend),
            "average_order_value": round(average_order_value_trend, 2),
            "customer_satisfaction": round(customer_satisfaction_trend, 1),
        }

        # Convert data to JSON
        json_data = json.dumps(mock_data)

        # Encrypt the JSON data
        encrypted_data = encrypt_data(json_data, username="WebSocketServer")

        print("Sending data:", json_data)
        await websocket.send(encrypted_data)

        # Wait for 1 second before sending the next update
        await asyncio.sleep(1)


async def receive_client_message(websocket, path):
    async for message in websocket:
        # Decrypt the received message
        decrypted_message = decrypt_data(message, username="WebSocketServer")
        print(f"Received from client: {decrypted_message}")


async def handle_connection(websocket, path):
    send_task = asyncio.create_task(send_mock_data(websocket, path))
    receive_task = asyncio.create_task(receive_client_message(websocket, path))
    await asyncio.gather(send_task, receive_task)


async def main():
    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("Mock WebSocket server started on ws://localhost:8765")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
