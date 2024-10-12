import os
import logging
from functools import wraps
from cryptography.fernet import Fernet, InvalidToken
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
def encrypt_data(data, username="Unknown"):
    """
    Encrypts the provided data using the global ENCRYPTION_KEY.

    Args:
        data (str): The data to be encrypted.
        username (str): The username of the person requesting encryption.

    Returns:
        str: The encrypted data encoded in a URL-safe format.
    """
    f = Fernet(ENCRYPTION_KEY)
    encrypted_data = f.encrypt(data.encode())
    result = base64.urlsafe_b64encode(encrypted_data).decode()

    logger.debug(f"Encryption - Encrypted data length: {len(encrypted_data)}")
    logger.debug(f"Encryption - Final result length: {len(result)}")

    return result


@log_function_call
def decrypt_data(encrypted_data, username="Unknown"):
    """
    Decrypts the provided encrypted data using the global ENCRYPTION_KEY.

    Args:
        encrypted_data (str): The encrypted data to be decrypted.
        username (str): The username of the person requesting decryption.

    Returns:
        str: The decrypted data.
    """
    try:
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode()

        encrypted_data = encrypted_data.strip()
        missing_padding = len(encrypted_data) % 4
        if missing_padding:
            encrypted_data += b"=" * (4 - missing_padding)

        decoded_data = base64.urlsafe_b64decode(encrypted_data)

        f = Fernet(ENCRYPTION_KEY)
        decrypted_data = f.decrypt(decoded_data)
        return decrypted_data.decode()

    except InvalidToken:
        logger.error("Invalid token error during decryption.")
    except Exception as e:
        logger.error(f"Unexpected error during decryption: {str(e)}")
    return None


@log_function_call
def hash_password(password):
    """
    Hashes the provided password using a derived key.

    Args:
        password (str): The password to be hashed.

    Returns:
        bytes: The salt and derived key.
    """
    salt = os.urandom(16)
    key = derive_key(password, salt)
    return salt + key


@log_function_call
def verify_password(stored_password, provided_password):
    """
    Verifies the provided password against the stored password.

    Args:
        stored_password (bytes): The stored password.
        provided_password (str): The password to verify.

    Returns:
        bool: True if the password matches, False otherwise.
    """
    salt = stored_password[:16]
    stored_key = stored_password[16:]
    derived_key = derive_key(provided_password, salt)
    return derived_key == stored_key


@log_function_call
def log_access(user, action):
    """
    Logs user access actions.

    Args:
        user (str): The user performing the action.
        action (str): The action performed by the user.
    """
    logging.info(f"User {user} performed action: {action}")


# Example of role-based access control (RBAC)
roles = {
    "admin": ["read", "write", "delete"],
    "user": ["read"],
    "analyst": ["read", "write"],
}


@log_function_call
def check_permission(user_role, action):
    """
    Checks if a user has permission to perform a specific action.

    Args:
        user_role (str): The role of the user.
        action (str): The action to check permission for.

    Returns:
        bool: True if the user has permission, False otherwise.
    """
    if user_role in roles and action in roles[user_role]:
        return True
    return False
