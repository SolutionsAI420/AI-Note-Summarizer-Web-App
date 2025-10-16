import os
from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import pipeline
import time
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import PyPDF2
from werkzeug.utils import secure_filename

# --- App and Model Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-super-secret-key-change-it'  # Important for sessions

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.ulqcmcvwscdqgndgrqgw:Y5soBbg9mmaEBg8o@aws-1-ap-south-1.pooler.supabase.com:6543/postgres'
db = SQLAlchemy(app)

# --- AI Models ---
models = {
    "Fast": pipeline("summarization", model="t5-small"),
    "High-Quality": pipeline("summarization", model="facebook/bart-large-cnn")
}

# --- Database User Model ---
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
    return User.query.get(int(user_id))

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
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
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
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

# --- PDF Upload Settings ---
ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Home Route with PDF Support ---
@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    selected_model = "High-Quality"

    if request.method == "POST":
        if not current_user.is_authenticated:
            flash('Please log in to use the summarizer.')
            return redirect(url_for('login'))

        selected_model = request.form.get("model", "High-Quality")
        text = request.form.get("text", "").strip()
        file = request.files.get("file")

        # Extract text from PDF if uploaded
        if file and allowed_file(file.filename):
            try:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            except Exception as e:
                flash(f"⚠️ Error reading PDF: {str(e)}")
                return redirect(url_for('home'))

        if not text:
            flash("⚠️ Please provide text or upload a PDF file")
        else:
            try:
                result = models[selected_model](text, max_length=150, min_length=30, do_sample=False)
                summary = result[0]['summary_text']
            except Exception as e:
                summary = f"⚠️ Something went wrong: {str(e)}"

    return render_template("index.html", summary=summary, selected_model=selected_model)

# --- Run App ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
