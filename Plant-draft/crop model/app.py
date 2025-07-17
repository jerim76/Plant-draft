import tensorflow as tf
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import json
from werkzeug.utils import secure_filename
import openai
from dotenv import load_dotenv
import secrets

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))  # Generate random secret key if not set
bcrypt = Bcrypt(app)

# Session configuration
app.permanent_session_lifetime = timedelta(minutes=30)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agritech.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuration for uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set OpenAI API key (still needed for detection and chatbot)
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in .env file")

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
try:
    model = tf.keras.models.load_model('tomato_model_final.keras')
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

# Define class names
class_names = ['healthy', 'leaf blight', 'leaf curl', 'Septoria Leaf Spot', 'verticulium wilt']

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    name = db.Column(db.String(100), nullable=False, default='User')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    is_user = db.Column(db.Boolean, default=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('messages', lazy=True))

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('posts', lazy=True))

# Create database tables and default System user
with app.app_context():
    db.create_all()
    # Create or get System user (for potential future use, e.g., alerts)
    system_user = User.query.filter_by(email='system@agritech.com').first()
    if not system_user:
        system_user = User(
            email='system@agritech.com',
            password=bcrypt.generate_password_hash('system_password').decode('utf-8'),
            name='System'
        )
        db.session.add(system_user)
        db.session.commit()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid")
    img = cv2.resize(img, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def get_ai_recommendation(disease, confidence):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert plant pathologist specializing in tomato diseases. Provide accurate, concise, and actionable information."},
                {"role": "user", "content": f"""
                A tomato plant has been diagnosed with {disease} (detected with {confidence:.0%} confidence).
                Provide a detailed analysis in JSON format with the following keys:
                - description: Brief description of the disease (max 50 words)
                - causes: Array of main causes (max 3 items, each under 15 words)
                - symptoms: Array of key symptoms (max 3 items, each under 15 words)
                - immediate_actions: Array of immediate actions (max 3 items, each under 15 words)
                - organic_treatments: Array of organic treatments (max 3 items, each under 15 words)
                - chemical_treatments: Array of chemical treatments (max 2 items, each under 15 words)
                - prevention_tips: Array of prevention tips (max 3 items, each under 15 words)
                Ensure all information is accurate and specific to tomato plants.
                """}
            ],
            temperature=0.7,
            max_tokens=600
        )
        content = response.choices[0].message.content
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        return json.loads(content.strip())
    except Exception as e:
        print(f"OpenAI error: {e}")
        return {
            "description": f"{disease} affects tomato plants, causing reduced yield.",
            "causes": ["Fungal infection", "Poor soil conditions", "Excess moisture"],
            "symptoms": ["Yellowing leaves", "Spots on leaves", "Wilting stems"],
            "immediate_actions": ["Remove infected leaves", "Isolate affected plants"],
            "organic_treatments": ["Neem oil spray", "Baking soda solution"],
            "chemical_treatments": ["Copper-based fungicide"],
            "prevention_tips": ["Improve air circulation", "Water at soil level", "Crop rotation"]
        }

def get_chatbot_response(user_message, conversation_history):
    try:
        messages = [
            {"role": "system", "content": "You are a plant health AI assistant specializing in agricultural issues."}
        ]
        for msg in conversation_history:
            messages.append({
                "role": "user" if msg.is_user else "assistant",
                "content": msg.content
            })
        messages.append({"role": "user", "content": user_message})
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return "Sorry, I couldn't process your request. Please try again."

# Decorator to check if user is logged in
def login_required(f):
    def wrap(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

# Generate CSRF token
@app.before_request
def generate_csrf():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)

@app.route('/')
def index():
    try:
        if 'user_id' in session:
            user = User.query.get(session['user_id'])
            if user:
                return render_template('index.html', view='home-view', user=user)
            else:
                session.pop('user_id', None)
                session.pop('user', None)
                flash('User not found, please log in again', 'error')
        return render_template('index.html', view='login-view')
    except Exception as e:
        print(f"Index error: {e}")
        flash('An error occurred, please try again', 'error')
        return render_template('index.html', view='login-view')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        csrf_token = request.form.get('csrf_token')
        
        if csrf_token != session.get('csrf_token'):
            flash('Invalid CSRF token', 'error')
            return render_template('index.html', view='login-view')
        
        if not email or not password:
            flash('Please enter both email and password', 'error')
            return render_template('index.html', view='login-view')
        
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session.permanent = True
            session['user_id'] = user.id
            session['user'] = {'email': user.email, 'name': user.name}
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
            return render_template('index.html', view='login-view')
    
    return render_template('index.html', view='login-view')

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    password = request.form.get('password')
    name = request.form.get('name', 'User')
    csrf_token = request.form.get('csrf_token')
    
    if csrf_token != session.get('csrf_token'):
        flash('Invalid CSRF token', 'error')
        return render_template('index.html', view='login-view')
    
    if not email or not password:
        flash('Email and password are required', 'error')
        return render_template('index.html', view='login-view')
    
    if User.query.filter_by(email=email).first():
        flash('Email already registered', 'error')
        return render_template('index.html', view='login-view')
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(email=email, password=hashed_password, name=name)
    db.session.add(new_user)
    db.session.commit()
    
    flash('Registration successful! Please login.', 'success')
    return render_template('index.html', view='login-view')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user', None)
    session.pop('csrf_token', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'url': f'/static/uploads/{filename}'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            img = preprocess_image(filepath)
            predictions = model.predict(img)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            
            # Get detailed recommendations from OpenAI
            details = get_ai_recommendation(predicted_class, confidence)
            
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'image_url': f'/static/uploads/{filename}',
                'details': details
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            try:
                os.remove(filepath)
            except:
                pass
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/detection')
@login_required
def detection():
    user = User.query.get(session['user_id'])
    return render_template('index.html', view='detection-view', user=user)

@app.route('/forum', methods=['GET', 'POST'])
@login_required
def forum():
    user = User.query.get(session['user_id'])
    posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template('index.html', view='forum-view', user=user, posts=posts)

@app.route('/new_post', methods=['POST'])
@login_required
def new_post():
    title = request.form.get('title')
    content = request.form.get('content')
    csrf_token = request.form.get('csrf_token')
    
    if csrf_token != session.get('csrf_token'):
        flash('Invalid CSRF token', 'error')
        return redirect(url_for('forum'))
    
    if not title or not content:
        flash('Title and content are required', 'error')
        return redirect(url_for('forum'))
    
    new_post = Post(
        user_id=session['user_id'],
        title=title,
        content=content
    )
    db.session.add(new_post)
    db.session.commit()
    
    flash('Post created successfully', 'success')
    return redirect(url_for('forum'))

@app.route('/chatbot', methods=['GET', 'POST'])
@login_required
def chatbot():
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        message_content = request.form.get('message')
        if message_content:
            user_message = Message(
                user_id=user.id,
                content=message_content,
                is_user=True
            )
            db.session.add(user_message)
            
            history = Message.query.filter_by(user_id=user.id).order_by(Message.timestamp.desc()).limit(10).all()[::-1]
            
            ai_response = get_chatbot_response(message_content, history)
            
            ai_message = Message(
                user_id=user.id,
                content=ai_response,
                is_user=False
            )
            db.session.add(ai_message)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'user_message': message_content,
                'ai_response': ai_response
            })
    
    messages = Message.query.filter_by(user_id=user.id).order_by(Message.timestamp.asc()).all()
    return render_template('index.html', view='chatbot-view', user=user, messages=messages)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)