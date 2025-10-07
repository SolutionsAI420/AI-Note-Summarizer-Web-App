import os
from flask import Flask, render_template, request
from transformers import pipeline
import time

# Create Flask app
app = Flask(__name__)

# Load summarization models once
models = {
    "Fast": pipeline("summarization", model="t5-small"),
    "High-Quality": pipeline("summarization", model="facebook/bart-large-cnn")
}

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    selected_model = "High-Quality"
    if request.method == "POST":
        text = request.form.get("text", "")
        selected_model = request.form.get("model", "High-Quality")
        if text.strip():
            try:
                # Optional: simulate loading delay
                time.sleep(0.5)
                result = models[selected_model](text, max_length=150, min_length=30, do_sample=False)
                summary = result[0]['summary_text']
            except Exception as e:
                summary = f"⚠️ Something went wrong: {str(e)}"
    return render_template("index.html", summary=summary, selected_model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
