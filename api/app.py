from flask import Flask, request, jsonify
import os
import datetime
import pickle
from PIL import Image
import numpy as np
import util

app = Flask(__name__)

# Directory to store database files
db_dir = './db'
if not os.path.exists(db_dir):
    os.mkdir(db_dir)

# Path to the log file
log_path = './log.txt'

@app.route('/', methods=['GET'])
def home():
    return 'Welcome to the Face Recognition API!'

@app.route('/clockin', methods=['POST'])
def login():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Read the image file
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    name = util.recognize(image, db_dir)

    if name in ['unknown_person', 'no_persons_found']:
        return jsonify({'message': 'Unknown user. Please register new user or try again.'}), 400
    else:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write('{},{},in\n'.format(name, current_time))
        return jsonify({'message': 'Welcome, {}. Time: {}'.format(name, current_time)}), 200

@app.route('/clockout', methods=['POST'])
def logout():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Read the image file
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    name = util.recognize(image, db_dir)

    if name in ['unknown_person', 'no_persons_found']:
        return jsonify({'message': 'Unknown user. Please register new user or try again.'}), 400
    else:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write('{},{},out\n'.format(name, current_time))
        return jsonify({'message': 'Goodbye, {}. Time: {}'.format(name, current_time)}), 200

@app.route('/register-face', methods=['POST'])
def register():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    name = request.form.get('name')
    if not name:
        return jsonify({'message': 'Name is required'}), 400

    # Read the image file
    image = Image.open(file.stream).convert('RGB')
    image = np.array(image)

    embeddings = util.get_face_embeddings(image)
    if embeddings is None:
        return jsonify({'message': 'No face detected'}), 400

    with open(os.path.join(db_dir, '{}.pickle'.format(name)), 'wb') as file:
        pickle.dump(embeddings, file)

    # Log the registration event
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write('{},{},created at {}\n'.format(name, current_time, current_time))

    return jsonify({'message': 'User was registered successfully!'}), 200

@app.route('/delete-face', methods=['POST'])
def delete():
    name = request.form.get('name')
    if not name:
        return jsonify({'message': 'Name is required'}), 400

    file_path = os.path.join(db_dir, '{}.pickle'.format(name))
    if os.path.exists(file_path):
        os.remove(file_path)
        # Log the deletion event
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, 'a') as f:
            f.write('{},{},deleted at {}\n'.format(name, current_time, current_time))
        return jsonify({'message': 'User {} was deleted successfully!'.format(name)}), 200
    else:
        return jsonify({'message': 'User {} not found.'.format(name)}), 404

if __name__ == "__main__":
    app.run(debug=True)