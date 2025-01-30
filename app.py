from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
import logging
import json
from agent import ask

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return jsonify({'message': 'Server is running'}), 200

@app.route('/api/chat', methods=['POST'])
def send_chat_message():
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            logger.error("No message provided in request")
            return jsonify({'error': 'No message provided'}), 400
            
        response = ask(message)
        print(response)
        return jsonify({'response':(response)}), 200
            
    except Exception as e:
        error_msg = f"Error processing chat message: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            'error': 'An error occurred',
            'response': "I apologize, but something went wrong. Please try again."
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=False)
