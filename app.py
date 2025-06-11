from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datetime
import pickle
from PIL import Image
import numpy as np
import util

app = Flask(__name__)

CORS(app)

# Directory to store database files
db_dir = './db'
if not os.path.exists(db_dir):
    os.mkdir(db_dir)

# Path to the log file
log_path = './log.txt'

@app.route('/', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'API flask is running'}), 200

@app.route('/register-face', methods=['POST'])
def register():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part', 'registered': False}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file', 'registered': False}), 400

    name = request.form.get('name')
    if not name:
        return jsonify({'message': 'Name is required', 'registered': False}), 400

    user_file = os.path.join(db_dir, '{}.pickle'.format(name))
    if os.path.exists(user_file):
        return jsonify({'message': 'User with this name already exists!', 'registered': False}), 409

    # Read the image file
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    embeddings = util.get_face_embeddings(image)
    if embeddings is None:
        return jsonify({'message': 'No face detected', 'registered': False}), 400

    with open(user_file, 'wb') as file:
        pickle.dump(embeddings, file)

    # Log the registration event
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write('{},{},created at {}\n'.format(name, current_time, current_time))

    return jsonify({'message': 'User was registered successfully!', 'registered': True}), 200

@app.route('/delete-face', methods=['POST'])
def delete():
    name = request.form.get('name')
    if not name:
        return jsonify({'message': 'Name is required', 'deleted': False}), 400

    file_path = os.path.join(db_dir, '{}.pickle'.format(name))
    if os.path.exists(file_path):
        os.remove(file_path)
        # Log the deletion event
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write('{},{},deleted at {}\n'.format(name, current_time, current_time))
        return jsonify({'message': 'User {} was deleted successfully!'.format(name), 'deleted': True}), 200
    else:
        return jsonify({'message': 'User {} not found.'.format(name), 'deleted': False}), 404

@app.route('/check-user', methods=['POST'])
def check_user():
    if 'file' not in request.files:
        return jsonify({'recognized': False, 'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'recognized': False, 'message': 'No selected file'}), 400

    # Read the image file
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    name, score = util.recognize(image, db_dir)
    percent = round(score * 100, 2)

    if name in ['unknown_person', 'no_persons_found']:
        return jsonify({'recognized': False, 'score': percent, 'message': 'Unknown user. Please register new user or try again.'}), 404
    else:
        return jsonify({'recognized': True, 'name': name, 'score': percent, 'message': 'User recognized'}), 200

@app.route('/update-face', methods=['POST'])
def update():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part', 'updated': False}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file', 'updated': False}), 400

    name = request.form.get('name')
    if not name:
        return jsonify({'message': 'Name is required', 'updated': False}), 400

    user_file = os.path.join(db_dir, '{}.pickle'.format(name))
    if not os.path.exists(user_file):
        return jsonify({'message': 'User not found!', 'updated': False}), 404

    # Read the image file
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    embeddings = util.get_face_embeddings(image)
    if embeddings is None:
        return jsonify({'message': 'No face detected', 'updated': False}), 400

    with open(user_file, 'wb') as file:
        pickle.dump(embeddings, file)

    # Log the update event
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write('{},{},updated at {}\n'.format(name, current_time, current_time))

    return jsonify({'message': 'User face was updated successfully!', 'updated': True}), 200

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=5000, debug=True)