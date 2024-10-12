import asyncio
import websockets
import json
import random
import datetime
from encryption_utils import encrypt_data, decrypt_data
from dotenv import load_dotenv
import os


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


async def send_mock_data(websocket, path):
    while True:
        # Generate mock business metrics
        mock_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sales": random.randint(1000, 5000),
            "customers": random.randint(50, 200),
            "average_order_value": round(random.uniform(50, 200), 2),
            "customer_satisfaction": round(random.uniform(3.5, 5.0), 1),
        }

        # Convert data to JSON
        json_data = json.dumps(mock_data)

        # Encrypt the JSON data
        encrypted_data = encrypt_data(json_data, username="WebSocketServer")

        # Send encrypted data
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
