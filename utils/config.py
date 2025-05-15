import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration constants
MODEL_NAME = "gpt2"
ADAPTER_NAME = "fine_tune/output"  # Fixed duplicate variable
TRAIN_DATA_PATH = "fine_tune/data/train.jsonl"
PEFT_SAVE_DIR = "fine_tune/output"

# Get API key from environment variables with fallback
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")