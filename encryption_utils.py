import os
import logging
from functools import wraps
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# Read the encryption key from the file
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

# Set up logging
logging.basicConfig(
    filename="security.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Decorator for logging function calls with more details
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract username from kwargs or use 'Unknown' if not provided
        username = kwargs.get("username", "Unknown")

        # Create a string representation of arguments, excluding 'username'
        arg_str = ", ".join(
            [repr(a) for a in args[1:]]
            + [f"{k}={v!r}" for k, v in kwargs.items() if k != "username"]
        )

        logging.info(f"User: {username} - Called {func.__name__}({arg_str})")
        result = func(*args, **kwargs)
        logging.info(
            f"User: {username} - {func.__name__} returned: {result.__class__.__name__}"
        )
        return result

    return wrapper


@log_function_call
def generate_key(username="Unknown"):
    return Fernet.generate_key()


@log_function_call
def derive_key(password, salt, username="Unknown"):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


@log_function_call
def encrypt_data(data, username="Unknown"):
    f = Fernet(ENCRYPTION_KEY)
    return f.encrypt(data.encode())


@log_function_call
def decrypt_data(encrypted_data, username="Unknown"):
    try:
        f = Fernet(ENCRYPTION_KEY)
        decrypted_data = f.decrypt(encrypted_data)
        # print(f"Debug: decrypted_data: {decrypted_data}")
        return decrypted_data.decode()
    except InvalidToken:
        logging.error(f"InvalidToken error when decrypting data for user {username}")
        print(f"Debug: ENCRYPTION_KEY used for decryption: {ENCRYPTION_KEY}")
        return None
    except Exception as e:
        logging.error(f"Error decrypting data for user {username}: {str(e)}")
        print(f"Debug: ENCRYPTION_KEY used for decryption: {ENCRYPTION_KEY}")
        return None


@log_function_call
def hash_password(password, username="Unknown"):
    salt = os.urandom(16)
    key = derive_key(password, salt, username=username)
    return salt + key


@log_function_call
def verify_password(stored_password, provided_password, username="Unknown"):
    salt = stored_password[:16]
    stored_key = stored_password[16:]
    derived_key = derive_key(provided_password, salt, username=username)
    return derived_key == stored_key


@log_function_call
def log_access(user, action, username="Unknown"):
    logging.info(f"User {user} performed action: {action}")


# Example of role-based access control (RBAC)
roles = {
    "admin": ["read", "write", "delete"],
    "user": ["read"],
    "analyst": ["read", "write"],
}


@log_function_call
def check_permission(user_role, action, username="Unknown"):
    if user_role in roles and action in roles[user_role]:
        return True
    return False
