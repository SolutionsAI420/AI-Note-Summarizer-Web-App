import os
from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import pipeline
import time
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy # --- ADDED: For database integration

# --- App and Model Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-super-secret-key-change-it' # Important for sessions

# --- NEW: Database Configuration ---
# IMPORTANT: Replace this with your actual connection string from a service like Supabase
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.ulqcmcvwscdqgndgrqgw:Y5soBbg9mmaEBg8o@aws-1-ap-south-1.pooler.supabase.com:6543/postgres'
db = SQLAlchemy(app) # Initialize the database object

# --- AI Models (No Changes) ---
models = {
    "Fast": pipeline("summarization", model="t5-small"),
    "High-Quality": pipeline("summarization", model="facebook/bart-large-cnn")
}

# --- MODIFIED: Database User Model ---
# This class now defines a table in your database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)


# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    # MODIFIED: Fetches the user from the database by their ID
    return User.query.get(int(user_id))


# --- MODIFIED: Authentication Routes (Using the Database) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # MODIFIED: Look for the user in the database
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # MODIFIED: Check if the username already exists in the database
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            # MODIFIED: Create a new user record and save it to the database
            new_user = User(username=username, password_hash=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


# --- Home Route (No Changes) ---
@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    selected_model = "High-Quality"
    if request.method == "POST":
        if not current_user.is_authenticated:
            flash('Please log in to use the summarizer.')
            return redirect(url_for('login'))
        
        text = request.form.get("text", "")
        selected_model = request.form.get("model", "High-Quality")
        if text.strip():
            try:
                result = models[selected_model](text, max_length=150, min_length=30, do_sample=False)
                summary = result[0]['summary_text']
            except Exception as e:
                summary = f"⚠️ Something went wrong: {str(e)}"

    return render_template("index.html", summary=summary, selected_model=selected_model)


if __name__ == "__main__":
    # ADDED: This block creates the database tables if they don't exist yet
    with app.app_context():
        db.create_all()
    app.run(debug=True)