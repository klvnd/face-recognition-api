import os
import pickle
import face_recognition

def get_face_embeddings(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        return None
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings[0] if face_encodings else None

def recognize(img, db_path):
    # it is assumed there will be at most 1 match in the db
    
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found', 0.0
    else:
        embeddings_unknown = embeddings_unknown[0]

    db_dir = sorted(os.listdir(db_path))

    best_score = 0.0
    best_name = 'unknown_person'
    for db_file in db_dir:
        path_ = os.path.join(db_path, db_file)
        with open(path_, 'rb') as file:
            embeddings = pickle.load(file)
        # Calculate distance and convert to similarity (the lower the distance, the higher the similarity)
        distance = face_recognition.face_distance([embeddings], embeddings_unknown)[0]
        similarity = max(0.0, 1.0 - distance)  # similarity in range [0, 1]
        if similarity > best_score:
            best_score = similarity
            best_name = db_file[:-7]

    if best_score > 0.6:  # you can adjust this threshold
        return best_name, best_score
    else:
        return 'unknown_person', best_score