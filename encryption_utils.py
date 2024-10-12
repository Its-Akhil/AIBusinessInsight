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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Generate a static key for testing
static_key = Fernet.generate_key()
logger.debug(f"Static key generated: {static_key}")


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


# Decorator for logging function calls
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        username = kwargs.get("username", "Unknown")
        arg_str = ", ".join(
            [repr(a) for a in args[1:]]
            + [f"{k}={v!r}" for k, v in kwargs.items() if k != "username"]
        )
        logger.info(f"User: {username} - Called {func.__name__}({arg_str})")
        result = func(*args, **kwargs)
        logger.info(
            f"User: {username} - {func.__name__} returned: {result.__class__.__name__}"
        )
        return result

    return wrapper


@log_function_call
def generate_key(username="Unknown"):
    return Fernet.generate_key()


@log_function_call
def derive_key(username, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(username.encode()))
    return key


@log_function_call
def encrypt_data(data, username):
    salt = os.urandom(16)
    key = derive_key(username, salt)
    f = Fernet(key)
    encrypted_data = f.encrypt(data.encode())
    result = base64.urlsafe_b64encode(salt + encrypted_data).decode()

    logger.debug(f"Encryption - Salt: {salt.hex()}")
    logger.debug(f"Encryption - Derived key: {key}")
    logger.debug(f"Encryption - Encrypted data length: {len(encrypted_data)}")
    logger.debug(f"Encryption - Final result length: {len(result)}")

    return result


@log_function_call
def decrypt_data(encrypted_data, username):
    try:
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode()

        encrypted_data = encrypted_data.strip()
        missing_padding = len(encrypted_data) % 4
        if missing_padding:
            encrypted_data += b"=" * (4 - missing_padding)

        decoded_data = base64.urlsafe_b64decode(encrypted_data)

        if len(decoded_data) < 16:
            raise ValueError("Decoded data is too short")

        salt, encrypted_message = decoded_data[:16], decoded_data[16:]
        key = derive_key(username, salt)

        logger.debug(f"Decryption - Salt: {salt.hex()}")
        logger.debug(f"Decryption - Derived key: {key}")
        logger.debug(f"Decryption - Encrypted message length: {len(encrypted_message)}")

        f = Fernet(key)
        try:
            decrypted_data = f.decrypt(encrypted_message)
            return decrypted_data.decode()
        except InvalidToken:
            logger.warning("Decryption with derived key failed, trying static key...")
            f_static = Fernet(static_key)
            decrypted_data = f_static.decrypt(encrypted_message)
            return decrypted_data.decode()

    except base64.binascii.Error as e:
        logger.error(f"Base64 decoding failed: {str(e)}")
        logger.error(f"Problematic data: {encrypted_data[:50]}...")
    except ValueError as e:
        logger.error(f"Value error during decryption: {str(e)}")
    except InvalidToken as e:
        print(f"Invalid token error: {str(e)}")
        print(f"Salt: {salt.hex()}")
        print(f"Derived key: {key}")
        print(f"Encrypted message length: {len(encrypted_message)}")
    except Exception as e:
        logger.error(f"Unexpected error during decryption: {str(e)}")
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
