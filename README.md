# Face Recognition API

A **Flask-based API** for **registering, recognizing, updating, and deleting faces** using the `face_recognition` library.  
The project is containerized with **Docker** for easy deployment and portability.

---

## Key Features

- **Register Face** — Register a new user and store their facial embeddings in `.pickle` format  
- **Check User** — Detect and recognize a face from an uploaded image  
- **Update Face** — Update an existing user’s face data  
- **Delete Face** — Remove a user’s face data from the database  
- **Logging** — Record all create, update, and delete events in a log file

## Tech Stack

- **Python 3.11**
- **Flask** – Web framework  
- **face_recognition** – Face detection and recognition library  
- **Pillow (PIL)** – Image processing  
- **NumPy** – Numerical data processing  
- **Docker** – Containerization

---

## Prerequisites

Before running the project, make sure you have:

1. **Python 3.11 or later** installed  
   - [Download Python](https://www.python.org/downloads/)
2. **pip** (Python package manager)
3. **Virtual environment module (venv)** (comes pre-installed with Python)
4. (Optional) Installed [cmake](https://cmake.org/download/) and system build tools if you encounter face_recognition installation issues.


## Installation and Setup (Run with venv)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/klvnd/face-recognition-api.git
cd face-recognition-api
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
```

### 3️⃣ Activate the Virtual Environment
On Windows:
``` bash
venv\Scripts\activate
```

On macOS/Linux:
``` bash
source venv/bin/activate
```

You should now see (venv) at the beginning of your terminal prompt.

### 4️⃣ Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5️⃣ Run the Application
```bash
python app.py
```

Your API should now be running locally at:
👉 http://localhost:5000

### 6️⃣ Deactivate the Virtual Environment (when done)
After finishing your work:
```bash
deactivate
```

---

# 🧾 API Endpoints

### 🔹 `GET /`
**Description:** Health check endpoint to verify if the API is running.

**Response:**
```json
{
  "status": "ok",
  "message": "API flask is running"
}
```

### 🔹 `POST /register-face`
**Description:** Description: Register a new user's face and store its embeddings.

**Form Data:**
<table>
  <tr>
    <th>Key</th>
    <th>Type</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>file</code></td>
    <td>image</td>
    <td>✅</td>
    <td>Face image file to upload</td>
  </tr>
  <tr>
    <td><code>name</code></td>
    <td>string</td>
    <td>✅</td>
    <td>Name to register</td>
  </tr>
</table>


**Success Response:**
```json
{
  "message": "User was registered successfully!",
  "registered": true
}
```

**Error Response:**
```json
{
  "message": "Face not detected or already exists!",
  "registered": false
}
```

### 🔹 `POST /check-user`
**Description:** Detect and recognize an existing user's face.

**Form Data:**
<table>
  <tr>
    <th>Key</th>
    <th>Type</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>file</code></td>
    <td>image</td>
    <td>✅</td>
    <td>Image file to check</td>
  </tr>
</table>

**Success Response:**
```json
{
  "recognized": true,
  "name": "kelvin",
  "score": 94.56,
  "message": "User recognized"
}
```

**Failure Response:**
```json
{
  "recognized": false,
  "message": "Unknown user. Please register new user or try again."
}
```

### ### 🔹 `POST /update-face`
**Description:** Update face data for an existing user.

**Form Data:**
<table>
  <tr>
    <th>Key</th>
    <th>Type</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>file</code></td>
    <td>image</td>
    <td>✅</td>
    <td>New face image</td>
  </tr>
  <tr>
    <td><code>name</code></td>
    <td>string</td>
    <td>✅</td>
    <td>Username to update</td>
  </tr>
</table>

**Success Response:**
```json
{
  "message": "User face was updated successfully!",
  "updated": true
}
```

**Error Response:**
```json
{
  "message": "User not found or face not detected!",
  "updated": false
}
```

### ### 🔹 `POST /delete-face`
**Description:** Delete a user's face data from the database.

**Form Data:**
<table>
  <tr>
    <th>Key</th>
    <th>Type</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>name</code></td>
    <td>string</td>
    <td>✅</td>
    <td>Username to delete</td>
  </tr>
</table>

**Success Response:**
```json
{
  "message": "User {name} was deleted successfully!",
  "deleted": true
}
```

**Error Response:**
```json
{
  "message": "User not found!",
  "deleted": false
}
```
