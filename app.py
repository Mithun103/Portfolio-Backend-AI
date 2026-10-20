import os
import logging

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

from agent import graph

# --- Logging ---
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Flask ---
app = Flask(__name__)

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "https://mithunms.netlify.app",
    ).split(",")
    if origin.strip()
]

CORS(app, origins=CORS_ORIGINS)

logger.info("Starting Flask app with Redis bypassed.")
logger.info("CORS enabled for: %s", CORS_ORIGINS)


# --- Health Check ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Server is running"}), 200


# --- Chat Endpoint ---
@app.route("/api/chat", methods=["POST"])
def send_chat_message():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON object"}), 400

    message = data.get("message")

    if not isinstance(message, str) or not message.strip():
        return jsonify({
            "error": "A non-empty 'message' string is required"
        }), 400

    # The frontend can still send session_id.
    # It is unused while conversation history is disabled.
    try:
        result = graph.invoke({
            "user_query": message.strip(),
            "chat_history": [],
        })

        response = result.get("response")

        if not isinstance(response, str):
            raise ValueError("Agent returned an invalid response")

        return jsonify({"response": response}), 200

    except Exception:
        logger.exception("Unexpected error during chat processing")
        return jsonify({
            "error": "Internal server error",
            "response": (
                "Oops! Something went wrong. Please try again later."
            ),
        }), 500
