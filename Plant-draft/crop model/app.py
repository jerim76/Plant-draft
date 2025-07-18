from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, session, Markup
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import openai
from datetime import datetime
import secrets
from functools import lru_cache
import re

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agritech.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
load_dotenv()

# OpenAI API setup
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define model class names with dots
MODEL_CLASSES = [
    'healthy','leaf curl','leaf blight','septoria leaf spot','verticulium wilt'
]

# Cache for OpenAI responses
@lru_cache(maxsize=100)
def get_openai_details(disease):
    if disease not in MODEL_CLASSES:
        return "Invalid disease prediction from model."
    print(f'Fetching OpenAI details for: {disease}')
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a plant pathology expert providing detailed information on tomato diseases. Only provide information for the exact disease name given, using the following classes with dots: Tomato___Bacterial.spot, Tomato___Early.blight, Tomato___healthy, Tomato___Late.blight, Tomato___Leaf.Mold, Tomato___Septoria.leaf.spot, Tomato___Spider.mites.Two.spotted.spider.mite, Tomato___Target.Spot, Tomato___Tomato.mosaic.virus, Tomato___Tomato.Yellow.Leaf.Curl.Virus. Ensure all sections (description, causes, symptoms, immediate actions, organic treatments, chemical treatments, prevention tips) are included, even if brief. Use your own formatting for lists or paragraphs as appropriate."},
                {"role": "user", "content": f"Provide detailed information about {disease}, including description, causes, symptoms, immediate actions, organic treatments, chemical treatments, and prevention tips. Do not infer or suggest other diseases; stick to the exact name provided."}
            ]
        )
        details = response.choices[0].message['content']
        print('OpenAI response received')
        return details
    except Exception as e:
        print(f'OpenAI API error: {str(e)}')
        return f"Failed to fetch details: {str(e)}"

# Clean Markdown and apply HTML bold
def clean_markdown(text):
    text = re.sub(r'#+\s?', '', text)  # Remove # and following spaces
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)  # Replace **bold** with <strong>bold</strong>
    return text

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    posts = db.relationship('Post', backref='user', lazy=True)
    community_posts = db.relationship('CommunityPost', backref='user', lazy=True)
    responses = db.relationship('CommunityResponse', backref='response_author', lazy=True)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    tags = db.Column(db.String(200), nullable=True)
    category = db.Column(db.String(50), nullable=True)  # New field
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class CommunityPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    image_path = db.Column(db.String(200))
    location = db.Column(db.String(100))
    category = db.Column(db.String(50), nullable=True)  # New field
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    responses = db.relationship('CommunityResponse', backref='post', lazy=True)

class CommunityResponse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('community_post.id'), nullable=False)
    user = db.relationship('User')  # Removed backref to avoid conflict

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
try:
    model = load_model('tomato_model_final.keras')
    print('Model loaded successfully')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Helper function for allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# CSRF token generation
@app.before_request
def generate_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(16)
    print('CSRF token:', session['csrf_token'])

# CSRF validation
def validate_csrf_token():
    token = request.headers.get('X-CSRF-Token') or request.form.get('csrf_token')
    print(f'Validating CSRF token: received={token}, expected={session.get("csrf_token")}')
    if not token or token != session.get('csrf_token'):
        print('CSRF validation failed')
        return False
    print('CSRF validation passed')
    return True

# Routes
@app.route('/')
def index():
    print('Index route called')
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
        return render_template('home.html', user=user, view='home')
    return render_template('login.html', view='login')

@app.route('/login', methods=['POST'])
def login():
    print('Login route called')
    if not validate_csrf_token():
        flash('CSRF token invalid', 'error')
        print('Login failed: Invalid CSRF token')
        return redirect(url_for('index'))
    
    try:
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            print(f'Login successful for user: {email}')
            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'error')
            print(f'Login failed for email: {email}')
            return redirect(url_for('index'))
    except Exception as e:
        print(f'Login error: {str(e)}')
        flash(f'Error during login: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/register', methods=['POST'])
def register():
    print('Register route called')
    if not validate_csrf_token():
        flash('CSRF token invalid', 'error')
        print('Registration failed: Invalid CSRF token')
        return redirect(url_for('index'))
    
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            print(f'Registration failed: Email {email} already registered')
            return redirect(url_for('index'))
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(name=name, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        flash('Registration successful!', 'success')
        print(f'Registration successful for user: {email}')
        return redirect(url_for('index'))
    except Exception as e:
        print(f'Registration error: {str(e)}')
        flash(f'Error during registration: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    print('Logout route called')
    session.pop('user_id', None)
    session.pop('chat_messages', None)
    session.pop('csrf_token', None)
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_file():
    print('Upload route called')
    if not validate_csrf_token():
        print('Upload failed: Invalid CSRF token')
        return jsonify({'error': 'Invalid CSRF token'}), 403
    
    if 'file' not in request.files:
        print('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        print('No file selected')
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f'File saved: {file_path}')
        return jsonify({'filename': filename, 'file_path': file_path}), 200
    print('Invalid file type')
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    print('Predict route called')
    if not validate_csrf_token():
        print('Predict failed: Invalid CSRF token')
        return jsonify({'error': 'Invalid CSRF token'}), 403
    
    if not model:
        print('Model not loaded')
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        print('No file part in request')
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        print('No file selected')
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f'File saved: {file_path}')
        
        try:
            img = cv2.imread(file_path)
            if img is None:
                print('Invalid image file')
                return jsonify({'error': 'Invalid image file'}), 400
            
            print('Processing image')
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            print('Making prediction')
            predictions = model.predict(img)
            predicted_class = MODEL_CLASSES[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            print(f'Prediction: {predicted_class}, Confidence: {confidence}')
            
            print('Requesting OpenAI details')
            details = get_openai_details(predicted_class)
            cleaned_details = clean_markdown(details)
            
            parsed_details = {
                'description': Markup(cleaned_details.split('Description:')[1].split('Causes:')[0].strip()) if 'Description:' in cleaned_details else Markup('No description available'),
                'causes': Markup(cleaned_details.split('Causes:')[1].split('Symptoms:')[0].strip()) if 'Causes:' in cleaned_details and 'Symptoms:' in cleaned_details else Markup('No causes available'),
                'symptoms': Markup(cleaned_details.split('Symptoms:')[1].split('Immediate Actions:')[0].strip()) if 'Symptoms:' in cleaned_details and 'Immediate Actions:' in cleaned_details else Markup('No symptoms available'),
                'immediate_actions': Markup(cleaned_details.split('Immediate Actions:')[1].split('Organic Treatments:')[0].strip()) if 'Immediate Actions:' in cleaned_details and 'Organic Treatments:' in cleaned_details else Markup('No immediate actions available'),
                'organic_treatments': Markup(cleaned_details.split('Organic Treatments:')[1].split('Chemical Treatments:')[0].strip()) if 'Organic Treatments:' in cleaned_details and 'Chemical Treatments:' in cleaned_details else Markup('No organic treatments available'),
                'chemical_treatments': Markup(cleaned_details.split('Chemical Treatments:')[1].split('Prevention Tips:')[0].strip()) if 'Chemical Treatments:' in cleaned_details and 'Prevention Tips:' in cleaned_details else Markup('No chemical treatments available'),
                'prevention_tips': Markup(cleaned_details.split('Prevention Tips:')[1].strip()) if 'Prevention Tips:' in cleaned_details else Markup('No prevention tips available')
            }
            
            print('Returning prediction response')
            return jsonify({
                'success': True,
                'prediction': predicted_class,
                'confidence': confidence,
                'image_url': f"/{file_path}",
                'details': parsed_details
            })
        except Exception as e:
            print(f'Prediction error: {str(e)}')
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    print('Invalid file type')
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/detection')
def detection():
    print('Detection route called')
    if 'user_id' not in session:
        flash('Please log in to access this page', 'error')
        print('User not logged in, redirecting to index')
        return redirect(url_for('index'))
    return render_template('detection.html', view='detection')

@app.route('/forum', methods=['GET', 'POST'])
def forum():
    print('Forum route called')
    if 'user_id' not in session:
        flash('Please log in to access the forum', 'error')
        print('User not logged in, redirecting to index')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        if not validate_csrf_token():
            print('Forum post failed: Invalid CSRF token')
            return jsonify({'error': 'Invalid CSRF token'}), 403
        
        title = request.form.get('title')
        content = request.form.get('content')
        tags = request.form.get('tags', '').strip()
        location = request.form.get('location')
        category = request.form.get('category', 'All')  # Default to 'All' if not provided
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
            else:
                file_path = None
        else:
            file_path = None
        
        if title and content:
            post = Post(title=title, content=content, tags=tags, category=category, user_id=session['user_id'])
            db.session.add(post)
            db.session.commit()
            if file_path or location:
                community_post = CommunityPost(content=content, image_path=file_path, location=location, category=category, user_id=session['user_id'])
                db.session.add(community_post)
                db.session.commit()
            flash('Post created successfully!', 'success')
            print(f'Post created: {title}, Tags: {tags}, Category: {category}, Location: {location}')
            return redirect(url_for('forum'))
        else:
            flash('Title and content are required', 'error')
            print('Forum post failed: Missing title or content')
    
    posts = Post.query.order_by(Post.created_at.desc()).all()
    community_posts = CommunityPost.query.order_by(CommunityPost.created_at.desc()).all()
    return render_template('forum.html', posts=posts, community_posts=community_posts, view='forum')

@app.route('/forum/response/<int:post_id>', methods=['POST'])
def forum_response(post_id):
    print(f'Forum response route called for post {post_id}')
    if 'user_id' not in session:
        flash('Please log in to respond', 'error')
        print('User not logged in, redirecting to forum')
        return redirect(url_for('forum'))
    
    if not validate_csrf_token():
        print('Forum response failed: Invalid CSRF token')
        return jsonify({'error': 'Invalid CSRF token'}), 403
    
    post = CommunityPost.query.get_or_404(post_id)
    content = request.form.get('response')
    
    if content:
        response = CommunityResponse(content=content, user_id=session['user_id'], post_id=post_id)
        db.session.add(response)
        db.session.commit()
        flash('Response added successfully!', 'success')
        print(f'Response added to post {post_id}')
    else:
        flash('Response content is required', 'error')
        print('Forum response failed: Missing content')
    
    return redirect(url_for('forum'))

@app.route('/new_post', methods=['POST'])
def new_post():
    print('New post route called')
    try:
        if not validate_csrf_token():
            print('New post failed: Invalid CSRF token')
            return jsonify({'error': 'Invalid CSRF token'}), 403
        
        if 'user_id' not in session:
            flash('Please log in to create a post', 'error')
            print('User not logged in, redirecting to index')
            return redirect(url_for('index'))
        
        title = request.form.get('title')
        content = request.form.get('content')
        tags = request.form.get('tags', '').strip()
        user_id = session['user_id']
        
        if title and content:
            post = Post(title=title, content=content, tags=tags, user_id=user_id)
            db.session.add(post)
            db.session.commit()
            flash('Post created successfully!', 'success')
            print(f'Post created: {title}, Tags: {tags}')
            return redirect(url_for('forum'))
        else:
            flash('Title and content are required', 'error')
            print('Post creation failed: Missing title or content')
            return redirect(url_for('index'))
    except Exception as e:
        print(f'New post error: {str(e)}')
        flash(f'Error creating post: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    print('Chatbot route called')
    try:
        if 'user_id' not in session:
            flash('Please log in to access the chatbot', 'error')
            print('User not logged in, redirecting to index')
            return redirect(url_for('index'))
        
        messages = session.get('chat_messages', [])
        
        if request.method == 'POST':
            if not validate_csrf_token():
                print('Chatbot failed: Invalid CSRF token')
                return jsonify({'error': 'Invalid CSRF token'}), 403
            
            user_message = request.form.get('message')
            print(f'Received chat message: {user_message}')
            if user_message:
                messages.append({'content': user_message, 'is_user': True})
                
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a plant health AI assistant. Provide accurate and helpful responses about plant care, diseases, and farming practices."},
                            {"role": "user", "content": user_message}
                        ]
                    )
                    ai_response = response.choices[0].message['content']
                    messages.append({'content': ai_response, 'is_user': False})
                    session['chat_messages'] = messages[-10:]  # Keep last 10 messages
                    print('Chatbot response sent')
                    return jsonify({'success': True, 'ai_response': ai_response})
                except Exception as e:
                    print(f'Chatbot error: {str(e)}')
                    return jsonify({'error': f'Chatbot error: {str(e)}'}), 500
        
        return render_template('chatbot.html', messages=messages, view='chatbot')
    except Exception as e:
        print(f'Chatbot route error: {str(e)}')
        flash(f'Error loading chatbot: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    try:
        with app.app_context():
            db.create_all()
            print("Database initialized with tables: user, post, community_post, community_response")
            if not User.query.first():
                email = "tweep1900@gmail.com"
                password = "testpassword"
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                test_user = User(name="Test User", email=email, password=hashed_password)
                db.session.add(test_user)
                db.session.commit()
                print(f"Added default test user: {email}")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
    app.run(debug=True)