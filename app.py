import os
import logging
import redis
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.chat_message_histories import RedisChatMessageHistory
from agent import personal_ai_agent  # Ensure this function is defined and imported correctly

# --- Load Environment Variables ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Flask Setup ---
app = Flask(__name__)

# --- Config ---
FLASK_ENV = os.getenv("FLASK_ENV", "production")
DEBUG_MODE = FLASK_ENV == "development"
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://default:oTkDq0Dfd3p0UZyKiRDgnWmx0PfjgpNX@redis-12966.c266.us-east-1-3.ec2.redns.redis-cloud.com:12966"
)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "https://mithunms.netlify.app").split(",")
CORS(app, origins=CORS_ORIGINS)

logger.info(f"🚀 Starting Flask app in {FLASK_ENV} mode.")
logger.info(f"✅ CORS enabled for: {CORS_ORIGINS}")
logger.info(f"🔗 Connecting to Redis: {REDIS_URL}")

# --- Redis Client Setup ---
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=False)
    redis_client.ping()
    logger.info("✅ Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")
    redis_client = None

# --- Health Check ---
@app.route('/')
def home():
    return jsonify({'message': '✅ Server is running'}), 200

# --- Main Chat Endpoint ---
@app.route('/api/chat', methods=['POST'])
def send_chat_message():
    if not redis_client:
        return jsonify({
            'error': 'Redis unavailable',
            'response': "I'm sorry, my memory is offline. Try again later."
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': "Invalid JSON"}), 400

        message = data.get('message')
        session_id = data.get('session_id')

        if not message or not session_id:
            return jsonify({'error': "Missing 'message' or 'session_id'"}), 400

        logger.info(f"📩 User message: {message} | Session: {session_id}")

        chat_history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
        result = personal_ai_agent(message, chat_history)

        # ✅ FIX: Ensure response is a string (not a nested dict)
        response = result.get("response")
        if isinstance(response, dict) and "response" in response:
            response = response["response"]

        logger.info(f"🤖 AI response: {response}")
        return jsonify({'response': response}), 200

    except Exception as e:
        logger.exception("❌ Unexpected error during chat processing")
        return jsonify({
            'error': 'Internal server error',
            'response': "Oops! Something went wrong. Please try again later."
        }), 500

# --- Note ---
# Use flask run or gunicorn to run this file
# Example (dev): FLASK_APP=app.py FLASK_ENV=development flask run
