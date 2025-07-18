from app import app, db, bcrypt, User
from datetime import datetime

with app.app_context():
    # Drop existing tables (optional, only if you want to start fresh)
    db.drop_all()
    print("Dropped all tables")
    
    # Create tables
    db.create_all()
    print("Created tables: user, post")
    
    # Add test user
    email = "tweep1900@gmail.com"
    password = "testpassword"
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    test_user = User(name="Test User", email=email, password=hashed_password)
    db.session.add(test_user)
    db.session.commit()
    print(f"Added test user: {email}")
    
    # Verify tables
    tables = db.engine.table_names()
    print(f"Tables in database: {tables}")
    
    # Verify user
    user = User.query.filter_by(email=email).first()
    print(f"Verified user: {user.email if user else 'Not found'}")