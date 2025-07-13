from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
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

@app.route('/')
def home():
    """A simple health check endpoint."""
    return jsonify({'message': 'Server is running'}), 200

@app.route('/api/chat', methods=['POST'])
def send_chat_message():
    """
    Receives a chat message, passes it to the AI agent,
    and returns the agent's response. This endpoint is stateless.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': "Invalid JSON"}), 400

        message = data.get('message')

        # Validate that a message was provided
        if not message:
            return jsonify({'error': "Missing 'message' in request body"}), 400

        # The agent is now called without memory.
        # This assumes the `personal_ai_agent` function can handle a single argument.
        result = personal_ai_agent(message)

        return jsonify({'response': result["response"]}), 200

    except Exception as e:
        logger.exception("Error during chat processing")
        return jsonify({
            'error': 'Internal server error',
            'response': "Oops, something went wrong. Please try again."
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Flask server...")
    # Use debug=False in a production environment
    app.run(debug=True)
