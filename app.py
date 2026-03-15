from flask import Flask, jsonify, send_from_directory
import os
import json

app = Flask(__name__)

@app.route('/')
def index():
    with open('index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/images')
def get_images():
    folder = 'samples'
    if not os.path.exists(folder):
        return jsonify([])
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    return jsonify(files)

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory('samples', filename)

@app.route('/logs')
def get_logs():
    path = 'logs/history.json'
    if not os.path.exists(path):
        return jsonify({"critic_loss": [], "gen_loss": []})
    with open(path, 'r', encoding='utf-8') as f:
        return jsonify(json.load(f))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
