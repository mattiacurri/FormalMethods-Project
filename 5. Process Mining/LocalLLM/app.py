import os
import requests
from flask import Flask, request, render_template, Response
import pandas as pd
import OllamaClient as OC
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return Response(stream_with_context(analyze_dataset(filepath)), 
                   mimetype='text/event-stream')

def generate_llm_response(model, current_prompt):
    """Generates a streaming response from Ollama based on the current state."""
    client = OC.OllamaClient()
    try:
        payload = {
            "prompt": current_prompt,
            "model": model,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 75,
            "options": {
                "num_ctx": 16384
            }
        }
        
        for token in client.generate_stream(payload):
            yield token
        
    except requests.exceptions.RequestException as e:
        yield f"Errore di connessione a Ollama: {e}"
    except Exception as e:
        yield f"Errore nella generazione della risposta: {e}"

def analyze_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        analysis_prompt = f"Here is an activity recognition dataset. Do a full analysis. Find any anomalies, bottlenecks, and special things: \n{df}"
        for token in generate_llm_response("llama3.2", analysis_prompt):
            yield f"{token}"
    except Exception as e:
        yield f"data: Error loading dataset: {e}\n\n"

def stream_with_context(generator):
    """Stream the response from the generator."""
    for line in generator:
        if line:
            yield f"{line}"

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
