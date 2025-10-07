from transformers import pipeline
import gradio as gr

# Load models
models = {
    "Fast": pipeline("summarization", model="t5-small"),
    "High-Quality": pipeline("summarization", model="facebook/bart-large-cnn")
}

# Summarization function
def summarize(text, model_name):
    if not text.strip():
        return "Please enter some text."
    result = models[model_name](text, max_length=150, min_length=30, do_sample=False)
    return result[0]['summary_text']

# Gradio Interface
iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=10, placeholder="Paste your notes here..."),
        gr.Dropdown(choices=["Fast", "High-Quality"], value="High-Quality", label="Choose Model")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="AI Note Summarizer",
    description="Paste your notes and get a summarized version instantly!"
)

iface.launch()
