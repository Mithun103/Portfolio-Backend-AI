import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory
from agent import personal_ai_agent

# --- Load Environment Variables ---
# This will load variables from a .env file for local development
load_dotenv()

# --- Setup Logging ---
# In production, you might want to configure this to output to a file or a logging service
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Flask App Setup ---
app = Flask(__name__)

# --- Configuration ---
# Load configuration from environment variables with sensible defaults
FLASK_ENV = os.getenv("FLASK_ENV", "production")
DEBUG_MODE = FLASK_ENV == "development"
REDIS_URL = os.getenv("REDIS_URL", "redis://default:oTkDq0Dfd3p0UZyKiRDgnWmx0PfjgpNX@redis-12966.c266.us-east-1-3.ec2.redns.redis-cloud.com:12966")

# Set up CORS based on environment variable, splitting by comma
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "https://mithunms.netlify.app").split(',')
CORS(app, origins=CORS_ORIGINS)

logger.info(f"Starting app in {FLASK_ENV} mode.")
logger.info(f"CORS enabled for: {CORS_ORIGINS}")
logger.info(f"Connecting to Redis at: {REDIS_URL}")


# --- Redis Connection ---
try:
    # Use the URL for a more flexible connection setup
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    logger.info("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"FATAL: Could not connect to Redis at {REDIS_URL}. The application cannot start without memory. Error: {e}")
    # In a real production setup, you might want the app to exit or have a retry mechanism.
    redis_client = None

@app.route('/')
def home():
    """A simple health check endpoint."""
    return jsonify({'message': 'Server is running'}), 200

@app.route('/api/chat', methods=['POST'])
def send_chat_message():
    """
    Receives a chat message and a session_id, uses Redis to manage
    conversation history, and returns the agent's response.
    """
    if not redis_client:
        return jsonify({
            'error': 'Internal server error: Memory system is offline',
            'response': "I'm sorry, but I can't remember our conversation right now because my memory system is offline."
        }), 503 # 503 Service Unavailable is more appropriate

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': "Invalid JSON"}), 400

        message = data.get('message')
        session_id = data.get('session_id')

        if not message or not session_id:
            return jsonify({'error': "Missing 'message' or 'session_id' in request body"}), 400

        chat_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        
        result = personal_ai_agent(message, chat_history)
        
        return jsonify({'response': result["response"]}), 200

    except Exception as e:
        logger.exception("An unexpected error occurred during chat processing") # More detailed log
        return jsonify({
            'error': 'Internal server error',
            'response': "Oops, something went wrong on our end. Please try again."
        }), 500

# The app.run() block is removed. The WSGI server will run the app.
# For local development, you can still run `flask run` after setting FLASK_APP=app.py
