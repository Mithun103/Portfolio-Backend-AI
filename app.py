# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from langchain.memory import ConversationBufferWindowMemory
from typing import Dict
from agent import personal_ai_agent

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app, origins=[
    "https://mithunms.netlify.app",
    "http://192.168.133.222:8080"
])

# In-memory store
session_memory_store: Dict[str, ConversationBufferWindowMemory] = {}
MEMORY_WINDOW_SIZE = 8

@app.route('/')
def home():
    return jsonify({'message': 'Server is running'}), 200

@app.route('/api/chat', methods=['POST'])
def send_chat_message():
    try:
        data = request.get_json()
        message = data.get('message')
        session_id = data.get('session_id')

        if not message or not session_id:
            return jsonify({'error': "Missing message or session_id"}), 400

        if session_id not in session_memory_store:
            session_memory_store[session_id] = ConversationBufferWindowMemory(
                k=MEMORY_WINDOW_SIZE,
                return_messages=True
            )

        memory = session_memory_store[session_id]
        result = personal_ai_agent(message, memory)

        return jsonify({'response': result["response"]}), 200

    except Exception as e:
        logger.exception("Error during chat processing")
        return jsonify({
            'error': 'Internal server error',
            'response': "Oops, something went wrong. Please try again."
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Flask server...")
    app.run(debug=True)
