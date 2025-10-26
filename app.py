from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import pipeline
import os, re
from collections import Counter
import yake
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from deep_translator import GoogleTranslator
from langdetect import detect
import PyPDF2
from werkzeug.utils import secure_filename

# ---------------------------
# Initialize Flask app
# ---------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-super-secret-key-change-it'
os.makedirs("static", exist_ok=True)

# ---------------------------
# Database Configuration
# ---------------------------
# NOTE: Replace with your actual database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres.ulqcmcvwscdqgndgrqgw:Y5soBbg9mmaEBg8o@aws-1-ap-south-1.pooler.supabase.com:6543/postgres'
db = SQLAlchemy(app)

# ---------------------------
# Flask-Login Setup
# ---------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Database User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- PDF Upload Settings ---
ALLOWED_EXTENSIONS = {'pdf'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# Lazy Loading Summarization Models 
# ---------------------------
@lru_cache(maxsize=2)
def get_model(name):
    """Load summarization models on demand and cache them."""
    if name == "Fast":
        print("⚙️ Loading T5-small summarization model...")
        return pipeline("summarization", model="t5-small")
    else:
        print("⚙️ Loading High-Quality summarization model (BART/T5-base)...")
        try:
            return pipeline("summarization", model="facebook/bart-large-cnn")
        except:
            return pipeline("summarization", model="t5-base")

# ---------------------------
# Lazy Loading NLP/Keyword Models (New)
# ---------------------------


@lru_cache(maxsize=1)
def get_nlp_models():
    """Light version: only YAKE and regex-based keyword extractor."""
    print("⚙️ Loading lightweight keyword model...")
    return {"kw_model": yake.KeywordExtractor(top=20, n=3), "embedder": SentenceTransformer('all-MiniLM-L6-v2')}

def hybrid_extract_keywords(text, top_n=12):
    """Simplified hybrid keyword extractor using YAKE + frequency filtering."""
    nlp_models = get_nlp_models()
    kw_model = nlp_models['kw_model']

    # Basic cleaning
    text_clean = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text_clean).lower().strip()
    if not cleaned:
        return []

    try:
        kw_yake = [kw for kw, _ in kw_model.extract_keywords(cleaned)]
    except Exception:
        kw_yake = []

    # Fallback: frequency-based keywords if YAKE fails
    if not kw_yake:
        words = re.findall(r'\b\w+\b', cleaned)
        freq = Counter(words)
        kw_yake = [w for w, _ in freq.most_common(top_n)]

    # Rank and filter
    freq = Counter(re.findall(r'\b\w+\b', cleaned))
    unique = []
    for k in kw_yake:
        if not any((k in u and k != u) or (u in k and k != u) for u in unique):
            unique.append(k)

    ranked = sorted(unique,
                    key=lambda k: sum(freq[w] for w in re.findall(r'\b\w+\b', k) if w in freq),
                    reverse=True)
    return ranked[:top_n]

# ---------------------------
# Semantic Mindmap Creation (Copied from previous step)
# ---------------------------
def create_humanlike_mindmap(text, summary, filename="static/mindmap.png"):
    """
    Creates a conceptual mind map (light version, no spaCy).
    """
    nlp_models = get_nlp_models()
    embedder = nlp_models['embedder']

    if not text.strip():
        return None

    source_text = text  # Use full text for better concepts
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', source_text.lower())

    # --- Improved Noun Phrase Extraction (1-3 words) ---
    words = cleaned_text.split()
    noun_phrases_clean = []
    for i in range(len(words)):
        for length in range(1, min(4, len(words) - i + 1)):
            phrase = " ".join(words[i:i+length])
            if re.match(r'^[a-z]+(?: [a-z]+){0,2}$', phrase):
                noun_phrases_clean.append(phrase)
    noun_phrases_clean = list(set(noun_phrases_clean))[:20]  # Cap for perf

    # Fallbacks
    if len(noun_phrases_clean) < 10:
        words_all = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', source_text.lower())]
        freq = Counter(words_all)
        main_topic = max((w for w in freq if len(w) > 4), key=freq.get, default="Main Topic").title()
        noun_phrases_clean = [w.title() for w in freq.most_common(15)]
        n_clusters = max(1, min(5, len(noun_phrases_clean) // 2))
    else:
        freq = Counter(re.findall(r'\b[a-zA-Z]{3,}\b', source_text.lower()))
        main_topic = max(noun_phrases_clean, key=lambda p: sum(freq[w] for w in p.split()))
        main_topic = main_topic.title()
        n_clusters = 5

    if len(noun_phrases_clean) < 2:
        return None

    embeddings = embedder.encode(noun_phrases_clean)

    # Cluster phrases
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    clusters = {}
    for label, phrase, embedding in zip(labels, noun_phrases_clean, embeddings):
        clusters.setdefault(label, []).append((phrase, embedding))

    # Frequency-based weighting from full text
    words_all = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
    freq_all = Counter(words_all)

    subtopics = []
    cluster_centroids = {label: np.mean([item[1] for item in items], axis=0) for label, items in clusters.items()}

    for label, group_items in clusters.items():
        if len(group_items) < 1:
            continue

        phrases = [item[0] for item in group_items]
        embeddings_group = np.array([item[1] for item in group_items])

        centroid_similarity = cosine_similarity(embeddings_group, cluster_centroids[label].reshape(1, -1)).flatten()
        subtopic_index = np.argmax(centroid_similarity)
        subtopic = ' '.join(phrases[subtopic_index].split()[:3]).title()

        details_candidates = [p for i, p in enumerate(phrases) if i != subtopic_index]
        ranked_details = sorted(details_candidates,
                                key=lambda x: sum(freq_all.get(w, 0) for w in re.findall(r'\b\w+\b', x.lower())),
                                reverse=True)
        details = [d.title() for d in ranked_details[:3]]

        if subtopic.lower() != main_topic.lower() and subtopic:
            subtopics.append((subtopic, details))

    # --- Mind Map Graph
    G = nx.Graph()
    G.add_node(main_topic, size=30000, color="#8ecae6")

    UNIFORM_EDGE_WIDTH = 4
    UNIFORM_EDGE_COLOR = '#444444'

    for sub, details in subtopics[:5]:  # Limit subtopics
        G.add_node(sub, size=20000, color="#90be6d")
        G.add_edge(main_topic, sub, width=UNIFORM_EDGE_WIDTH, color=UNIFORM_EDGE_COLOR)
        for det in details[:2]:  # Limit details per sub
            G.add_node(det, size=10000, color="#f9c74f")
            G.add_edge(sub, det, width=UNIFORM_EDGE_WIDTH, color=UNIFORM_EDGE_COLOR)

    if len(G.nodes) > 1:
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=1.0, iterations=100, seed=42)

        plt.figure(figsize=(20, 16))
        sizes = [G.nodes[n]['size'] for n in G.nodes]
        colors = [G.nodes[n]['color'] for n in G.nodes]

        labels = {}
        for n in G.nodes:
            short_n = n if len(n) <= 12 else n[:10] + '...'
            labels[n] = short_n.replace(' ', '\n')

        nx.draw(G, pos, with_labels=True, node_size=sizes, node_color=colors,
                font_size=14, font_weight='bold',
                edge_color=UNIFORM_EDGE_COLOR,
                width=UNIFORM_EDGE_WIDTH,
                alpha=0.9, labels=labels, arrows=False, node_shape='o')

        plt.title(f"Concept Mind Map - {main_topic}", fontsize=20, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)  # Lower DPI for speed
        plt.close()
        return filename
    return None

# ---------------------------
# Authentication Routes
# ---------------------------
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

# ---------------------------
# Main Route: Text/PDF Summarization with Multilingual Support (INTEGRATED)
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    keywords = []
    mindmap_path = None
    selected_model = "High-Quality"
    detected_lang = "en"
    text_input = ""

    if request.method == "POST":
        if not current_user.is_authenticated:
            flash('Please log in to use the summarizer.')
            return redirect(url_for('login'))

        action = request.form.get("action", "summarize")  # Default to summarize

        text_input = request.form.get("text", "").strip()
        file = request.files.get("file")
        selected_model = request.form.get("model", "High-Quality")
        target_lang = request.form.get("target_lang", "auto")

        # Clear old map only if mindmap action
        if action == "mindmap":
            if os.path.exists("static/mindmap.png"):
                os.remove("static/mindmap.png")

        # --- Extract text from PDF if uploaded ---
        if file and allowed_file(file.filename):
            try:
                reader = PyPDF2.PdfReader(file)
                text_input = ""
                for page in reader.pages:
                    text_input += page.extract_text() + "\n"
            except Exception as e:
                flash(f"⚠️ Error reading PDF: {str(e)}")
                return redirect(url_for('home'))

        if not text_input:
            flash("⚠️ Please provide text or upload a PDF file.")
        else:
            try:
                # --- 1. Detect input language ---
                detected_lang = detect(text_input)
                print(f"🌍 Detected language: {detected_lang}")

                # --- 2. Translate to English for NLP/Summarization ---
                if detected_lang != "en":
                    translated_text = GoogleTranslator(source=detected_lang, target='en').translate(text_input)
                else:
                    translated_text = text_input

                summary_english = ""
                keywords = []
                mindmap_path = None

                # --- 3. Summarize (on English text) only if action requires it ---
                if action == "summarize":
                    model = get_model(selected_model)
                    input_text = translated_text
                    if "t5" in model.model.config.name_or_path.lower():
                        input_text = f"summarize: {translated_text}"
                    if len(translated_text.split()) < 10:
                        summary_english = translated_text[:150] + "..."
                    else:
                        result = model(input_text, max_length=150, min_length=30, do_sample=False)
                        summary_english = result[0]['summary_text']

                # --- 4. EXTRACT KEYWORDS only if action requires it ---
                if action == "keywords":
                    keywords = hybrid_extract_keywords(translated_text)
                
                # --- 5. CREATE MIND MAP only if action requires it ---
                if action == "mindmap":
                    mindmap_path = create_humanlike_mindmap(translated_text, summary_english)

                # --- 6. Translate summary back ---
                final_target_lang = target_lang if target_lang != "auto" else detected_lang
                
                if final_target_lang != "en" and summary_english:
                    summary = GoogleTranslator(source='en', target=final_target_lang).translate(summary_english)
                    detected_lang = final_target_lang
                else:
                    summary = summary_english
                    detected_lang = "en"

            except Exception as e:
                summary = f"⚠️ Something went wrong: {str(e)}"
                
    return render_template("index.html",
                            summary=summary,
                            keywords=keywords,
                            mindmap_path=mindmap_path,
                            selected_model=selected_model,
                            detected_lang=detected_lang)
    
# ---------------------------
# Run the App
# ---------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(use_reloader=True, threaded=True, port=5000)