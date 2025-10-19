from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import pipeline
from keybert import KeyBERT
import spacy
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
    """Load KeyBERT, Spacy, and SentenceTransformer on demand."""
    print("⚙️ Loading NLP/Keyword models...")
    return {
        "kw_model": KeyBERT('all-MiniLM-L6-v2'),
        "nlp": spacy.load("en_core_web_sm"),
        "embedder": SentenceTransformer('all-MiniLM-L6-v2')
    }

# ---------------------------
# Hybrid Keyword Extraction (Copied from previous step)
# ---------------------------
def hybrid_extract_keywords(text, top_n=12):
    """Combine KeyBERT (prioritizing 1-2 words), YAKE, and linguistic filtering."""
    nlp_models = get_nlp_models()
    kw_model = nlp_models['kw_model']
    nlp = nlp_models['nlp']

    text_clean = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text_clean).lower().strip()
    if not cleaned:
        return []

    try:
        kw_bert_short = [kw for kw, score in kw_model.extract_keywords(
            cleaned, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=15, use_mmr=True, diversity=0.7)]
        kw_bert_long = [kw for kw, score in kw_model.extract_keywords(
            cleaned, keyphrase_ngram_range=(3, 3), stop_words='english', top_n=5, use_mmr=True, diversity=0.8)]
        kw_bert = kw_bert_short + [k for k in kw_bert_long if k not in kw_bert_short]
    except Exception:
        kw_bert = []

    try:
        kw_yake = [kw for kw, _ in yake.KeywordExtractor(top=15, n=3).extract_keywords(cleaned)]
    except Exception:
        kw_yake = []

    combined = list(dict.fromkeys(kw_bert + kw_yake))
    doc = nlp(text)
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    
    filtered = [kw for kw in combined if any(p in kw for p in noun_chunks) and len(kw.split()) <= 3]

    if not filtered:
        filtered = [kw for kw in kw_bert_short if len(kw.split()) <= 2][:top_n]
        if not filtered:
             filtered = combined[:top_n]

    freq = Counter(re.findall(r'\b\w+\b', cleaned))
    unique = []
    for k in filtered:
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
    Creates a human-like conceptual mind map with uniform edges.
    """
    nlp_models = get_nlp_models()
    nlp = nlp_models['nlp']
    embedder = nlp_models['embedder']
    
    if not text.strip():
        return None

    doc = nlp(summary if summary else text)
    source_text = summary if summary else text

    # Extract meaningful phrases (NPs - Max 3 words for mind map clarity)
    noun_phrases = [chunk.text.strip() for chunk in doc.noun_chunks if 
                    1 <= len(chunk.text.split()) <= 3 and len(chunk.text) > 3] 
    noun_phrases_clean = list(set([re.sub(r'[^a-zA-Z\s]', '', phrase.lower()).strip() for phrase in noun_phrases]))
    noun_phrases_clean = [p for p in noun_phrases_clean if p] 

    # Fallback/Clustering setup
    if len(noun_phrases_clean) < 5:
        words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', source_text)]
        filtered_words = [w for w in words if w not in nlp.Defaults.stop_words]
        main_topic = max(Counter(filtered_words), key=Counter(filtered_words).get, default="Main Topic").title()
        if len(noun_phrases_clean) < 3:
            noun_phrases_clean = list(set(filtered_words))[:10]
        n_clusters = max(1, min(5, len(noun_phrases_clean)))
    else:
        n_clusters = 5

    if not noun_phrases_clean: return None

    embeddings = embedder.encode(noun_phrases_clean)
    
    # Select main topic
    noun_phrases_sorted = sorted(
        [chunk.text.strip() for chunk in doc.noun_chunks if 1 <= len(chunk.text.split()) <= 3],
        key=lambda x: source_text.lower().count(x.lower()),
        reverse=True
    )
    main_topic = noun_phrases_sorted[0].title() if noun_phrases_sorted else "Main Concept"

    # Cluster phrases
    labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)

    clusters = {}
    for label, phrase, embedding in zip(labels, noun_phrases_clean, embeddings):
        clusters.setdefault(label, []).append((phrase, embedding))

    # Compute frequency for detail ranking
    words_all = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', text)]
    freq_all = Counter([w for w in words_all if w not in nlp.Defaults.stop_words])

    subtopics = []
    cluster_centroids = {label: np.mean([item[1] for item in items], axis=0) for label, items in clusters.items()}

    for label, group_items in clusters.items():
        if len(group_items) == 0: continue
        
        phrases = [item[0] for item in group_items]
        embeddings_group = np.array([item[1] for item in group_items])
        
        centroid_similarity = cosine_similarity(embeddings_group, cluster_centroids[label].reshape(1, -1)).flatten()
        subtopic_index = np.argmax(centroid_similarity)
        subtopic = ' '.join(phrases[subtopic_index].split()[:3]) 
        
        details_candidates = [p for i, p in enumerate(phrases) if i != subtopic_index]
        ranked_details = sorted(details_candidates,
                                key=lambda x: sum(freq_all.get(w, 0) for w in re.findall(r'\b\w+\b', x)),
                                reverse=True)
        details = [' '.join(d.split()[:3]) for d in ranked_details[:3]] 

        if subtopic.lower() != main_topic.lower() and len(subtopic.split()) >= 1:
            subtopics.append((subtopic.title(), [d.title() for d in details]))

    # Build graph (Undirected for radial/spring layout)
    G = nx.Graph() 
    
    # NODE SIZES:
    G.add_node(main_topic, size=30000, color="#8ecae6") 
    
    # --- UNIFORM EDGE CONFIGURATION ---
    UNIFORM_EDGE_WIDTH = 4 
    UNIFORM_EDGE_COLOR = '#444444' 

    for sub, details in subtopics:
        G.add_node(sub, size=20000, color="#90be6d") 
        G.add_edge(main_topic, sub) 

        for det in details:
            det_name = f"{det}"
            G.add_node(det_name, size=10000, color="#f9c74f") 
            G.add_edge(sub, det_name) 

    # --- Layout for Radial Mindmap Look ---
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception as e:
        print(f"Kamada-Kawai layout failed: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(G, k=1.0, iterations=100)
        
    # --- Drawing ---
    plt.figure(figsize=(26, 20)) 

    sizes = [G.nodes[n]['size'] for n in G.nodes]
    colors = [G.nodes[n]['color'] for n in G.nodes]
    
    # Prepare labels: apply newline formatting for better spacing
    labels = {}
    for n in G.nodes:
        if len(n) > 15 or len(n.split()) > 2:
            labels[n] = n.replace(' ', '\n')
        else:
            labels[n] = n
            
    # Draw with uniform edge width and color
    nx.draw(G, pos, with_labels=True, node_size=sizes, node_color=colors,
            font_size=18, font_weight='bold', 
            edge_color=UNIFORM_EDGE_COLOR, 
            width=UNIFORM_EDGE_WIDTH,
            alpha=0.9, labels=labels, arrows=False, node_shape='o')

    plt.title(f"Concept Mind Map - {main_topic}", fontsize=24, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=220)
    plt.close()
    return filename


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

        text_input = request.form.get("text", "").strip()
        file = request.files.get("file")
        selected_model = request.form.get("model", "High-Quality")
        target_lang = request.form.get("target_lang", "auto")

        # Clear old map
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

                # --- 3. Summarize (on English text) ---
                model = get_model(selected_model)
                result = model(translated_text, max_length=150, min_length=30, do_sample=False)
                summary_english = result[0]['summary_text']
                
                # --- 4. EXTRACT KEYWORDS & CREATE MIND MAP (on English text) ---
                if len(translated_text) > 100: # Only run resource-intensive tasks on substantial text
                    keywords = hybrid_extract_keywords(translated_text)
                    mindmap_path = create_humanlike_mindmap(translated_text, summary_english)

                # --- 5. Translate summary back ---
                final_target_lang = target_lang if target_lang != "auto" else detected_lang
                
                if final_target_lang != "en":
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
    app.run(debug=False, use_reloader=False, threaded=False)
