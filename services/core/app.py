import requests
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure service URLs
ENCODER_URL = "http://127.0.0.1:5000/encode"
FAISS_URL = "http://127.0.0.1:5001/search"


@app.route("/")
def index():
    return render_template("search.html")


@app.route("/search", methods=["POST"])
def search():
    try:
        query_text = request.json["query"]  # type: ignore[no-untyped-call]

        # Get embedding from encoder
        encoder_response = requests.post(
            ENCODER_URL, json={"query": query_text}, timeout=10
        )
        encoder_response.raise_for_status()

        embedding = encoder_response.json()["embedding"]

        # Search FAISS
        faiss_response = requests.post(
            FAISS_URL,
            json={"type": "clip-laion", "embedding": embedding, "k": 3},
            timeout=15,
        )
        faiss_response.raise_for_status()

        # Ensure we always return an array of results
        results = faiss_response.json()
        if not isinstance(results, list):
            results = [results] if results else []

        return jsonify(results), 200

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Service error: {str(e)}"}), 500
    except KeyError as e:
        return jsonify({"error": f"Missing key: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
