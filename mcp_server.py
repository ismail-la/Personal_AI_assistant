from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/read', methods=['POST'])
def read_file():
    path = request.json.get('path')
    with open(path, 'r') as f:
        return jsonify({'content': f.read()})

@app.route('/run_chat', methods=['POST'])
def run_chat():
    prompt = request.json.get('prompt')
    # call FastAPI endpoint
    import requests
    resp = requests.post('http://127.0.0.1:8000/chat/', json={'prompt': prompt})
    return jsonify(resp.json())

if __name__ == '__main__':
    app.run(port=6000)