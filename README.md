
# 🌿 PlantGuardian – AI-Powered Plant Disease Detection

## 📘 Overview

**PlantGuardian** is an AI-driven platform designed to detect plant diseases through advanced computer vision techniques. Built on the Flask framework, it empowers farmers, agronomists, and gardeners with accurate, real-time diagnostics and treatment recommendations via an intuitive web interface.

---

## 📚 Table of Contents

1. [Overview](#overview)  
2. [Key Features](#key-features)  
3. [Technology Stack](#technology-stack)  
4. [Installation Guide](#installation-guide)  
5. [Usage Guide](#usage-guide)  
6. [API Documentation](#api-documentation)  
7. [Project Structure](#project-structure)  
8. [Contribution Guidelines](#contribution-guidelines)  
9. [License](#license)  
10. [Acknowledgements](#acknowledgements)  

---

## ✨ Key Features

### 🔬 Intelligent Plant Health Monitoring
- **AI-Powered Detection**: Identifies plant diseases using custom CNNs and transfer learning techniques.  
- **Visual Heatmaps**: Highlights affected leaf areas for better understanding.  
- **Treatment Advisory**: Recommends precise, data-driven interventions.  
- **Health Index Scoring**: Assesses plant vitality on a scale.  
- **Progress Tracking**: Maintains historical records for plant health trends.

### 📱 Seamless User Experience
- Image upload via drag-and-drop or file picker  
- Mobile-first, responsive design for field usability  
- Real-time feedback and results display  
- Multi-language support for accessibility  

---

## 🧰 Technology Stack

### Frontend
- **Languages**: HTML5, CSS3, JavaScript  
- **Styling**: Tailwind CSS  
- **Visualization**: Chart.js  
- **Machine Learning Preview**: TensorFlow.js  

### Backend
- **Framework**: Python Flask  
- **Machine Learning**: TensorFlow, Keras  
- **Image Processing**: OpenCV  
- **ORM**: SQLAlchemy  

### AI/ML Models
- Custom CNNs for classification  
- EfficientNetB3 (transfer learning)  
- Image segmentation for infection mapping  
- Confidence scoring for output reliability  

### Deployment
- Docker containerization  
- Gunicorn for production serving  
- Nginx reverse proxy  
- Hosting: AWS EC2 / Heroku  

---

## 🛠 Installation Guide

### Prerequisites
- Python 3.10+  
- PostgreSQL 14+  
- Node.js (for frontend builds)

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/your-username/plantguardian.git
cd plantguardian

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd static
npm install

# Set environment variables
cp .env.example .env
# (Update values in the .env file accordingly)

# Initialize and migrate database
flask db init
flask db migrate
flask db upgrade

# Download pretrained models (see models/README.md)

# Launch the development server
flask run
````

Access the application at: [http://localhost:5000](http://localhost:5000)

---

## 🚀 Usage Guide

### 🌿 Performing Disease Analysis

1. Navigate to the **Plant Analysis** section
2. Upload an image or use camera capture
3. Review diagnostic results, including:

   * Predicted disease
   * Confidence score
   * Treatment plan
4. Save and export analysis reports

### 📊 Dashboard Functionalities

* Review previous analyses
* Visualize health trends over time
* Compare diagnoses
* Export PDF reports

---

## 📡 API Documentation

The RESTful API adheres to the **OpenAPI 3.0 specification**. Interactive documentation is available at:
🔗 [http://localhost:5000/api-docs](http://localhost:5000/api-docs)

### Endpoints Summary

| Endpoint              | Method | Description                  | Parameters      |
| --------------------- | ------ | ---------------------------- | --------------- |
| `/api/plant/analysis` | POST   | Analyze uploaded plant image | `image` (file)  |
| `/api/plant/history`  | GET    | Retrieve analysis history    | `user_id` (int) |
| `/api/plant/report`   | GET    | Generate downloadable report | `analysis_id`   |
| `/api/plant/species`  | GET    | List supported plant species | None            |

### Sample Request (Python)

```python
import requests

url = "http://localhost:5000/api/plant/analysis"
files = {'image': open('leaf.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Sample JSON Response:**

```json
{
  "status": "success",
  "analysis": {
    "disease": "Tomato Early Blight",
    "confidence": 0.92,
    "health_score": 65,
    "treatments": [
      "Apply copper-based fungicide every 7-10 days",
      "Remove infected leaves immediately",
      "Improve air circulation around plants"
    ],
    "visual_analysis": "base64_encoded_image"
  }
}
```

---

## 🗂 Project Structure

```
plantguardian/
├── app/
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   ├── services/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── images/
│   │   └── node_modules/
│   ├── templates/
│   ├── __init__.py
│   └── config.py
├── ai_models/
│   └── plant_detection/
│       ├── model.h5
│       ├── classes.json
│       └── preprocess.py
├── migrations/
├── tests/
├── venv/
├── .env
├── .flaskenv
├── config.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🤝 Contribution Guidelines

🚧 This project is currently under a closed development phase.
📩 For collaboration or partnership inquiries, please contact the project maintainers directly.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0**.
Refer to the [LICENSE](LICENSE) file for full legal terms.

---

## 🙏 Acknowledgements

We gratefully acknowledge the contributions and support from:

* Open-source communities of **OpenCV** and **TensorFlow**
* Agricultural research institutions for model validation
* Farmers and beta testers for real-world feedback
* Developers and dataset providers of plant pathology corpora






