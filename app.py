from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from agent import ask

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ✅ Allow only your Netlify frontend to access the API
CORS(app, origins=[
    "https://mithunms.netlify.app",
    "http://192.168.133.222:8080"
])

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
        return jsonify({'response': response}), 200

    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        return jsonify({
            'error': 'An error occurred',
            'response': "I apologize, but something went wrong. Please try again."
        }), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True)
