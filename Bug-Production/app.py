from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
import code_analyzer
import code_analyze_ai
from collections import defaultdict
import requests
from urllib.parse import urlparse
import chardet
import nbformat
import gc
import re

app = Flask(__name__)

config_file_location = "config/dev.cfg"

if config_file_location:
    app.config.from_pyfile(config_file_location, silent=False)
else:
    app.config.from_pyfile('config/dev.cfg', silent=True)
    app.config.from_pyfile('config/prp.cfg', silent=True)
    app.config.from_pyfile('config/prod.cfg', silent=False)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['DOWNLOAD_FOLDER']):
    os.makedirs(app.config['DOWNLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


model = joblib.load('./saved_models/mk1plus.pkl')

scaler = joblib.load('./scaler/scaler.pkl')


def extract_code_from_ipynb(file_path, encoding):
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            nb = nbformat.read(f, as_version=4)
        code_cells = [cell['source'] for cell in nb.cells if cell['cell_type'] == 'code']
        return '\n\n'.join(code_cells)
    except Exception as e:
        return f"ipynb file reading error: {str(e)}"


def extract_metrics_from_file(file_path):
    try:
        rawdata = open(file_path, 'rb').read()
        result = chardet.detect(rawdata)
        encoding = result['encoding']

        extension = file_path.rsplit('.', 1)[1].lower()
        if extension == 'ipynb':
            code = extract_code_from_ipynb(file_path, encoding)
            if isinstance(code, str) and code.startswith("ipynb reading file error!"):
                return code
        else:
            with open(file_path, "r", encoding=encoding) as f:
                code = f.read()

    except (UnicodeDecodeError, Exception) as e:
        print(f"file reading error!: {e}")
        return None

    analyzer = code_analyzer.CodeAnalyzer()

    try:
        analyzer.analyze(code)
    except IndentationError as e:
        return f"There is an indentation error in your code. Please check it. Error: {str(e)}"
    except SyntaxError as e:
        return f"There is a syntax error in your code. Please check it. Error: {str(e)}"
    except Exception as e:
        return f"An unknown error occurred. Please check your code. Error: {str(e)}"
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
            gc.collect()

    metrics_dict = analyzer.get_metrics()

    metrics_dict["fanIn"] = sum(metrics_dict["fanIn"].values()) if isinstance(metrics_dict["fanIn"], defaultdict) else \
        metrics_dict["fanIn"]
    metrics_dict["fanOut"] = sum(
        len(v) if hasattr(v, "__len__") else 1 for v in metrics_dict["fanOut"].values()) if isinstance(
        metrics_dict["fanOut"], defaultdict) else metrics_dict["fanOut"]

    return pd.DataFrame([metrics_dict])


def analyze_with_ai(file_path, code):
    try:
        analysis, rating = code_analyze_ai.analyze_code(file_path)
        prompt = f"""
        project's code:
            {code}
          
        analysis result metrics:
            {analysis}
        
        Can you evaluate the efficiency and performance of the project and make suggestions?
        """
        ai_suggestions = code_analyze_ai.generate_suggestion(prompt)

        return analysis, ai_suggestions, rating

    except Exception as e:
        print(f"error occurred: {str(e)}")
        return None, None


def download_github_file(github_url, download_folder="downloads"):
    if github_url.startswith("https://github.com"):
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

        parsed_url = urlparse(github_url)
        file_name = os.path.basename(parsed_url.path)

        os.makedirs(download_folder, exist_ok=True)

        file_path = os.path.join(download_folder, file_name)

        response = requests.get(raw_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"File downloaded successfully: {file_path}")
            return file_path
        else:
            print(f"Failed to download the file. HTTP Status Code: {response.status_code}")
            return None
    else:
        print("Invalid GitHub URL provided.")
        return None


def encoding_detect(file_path):
    rawdata = open(file_path, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    with open(file_path, "r", encoding=encoding) as f:
        code = f.read()

    return code


def parse_ai_suggestions(suggestion_text):
    numbered_pattern = re.compile(
        r'^\s*(\d+)\.\s*\*\*([^*]+)\*\*\s*(.+?)(?=^\s*\d+\.\s*\*\*|\Z)',
        re.MULTILINE | re.DOTALL
    )

    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)

    suggestions = []
    matches = numbered_pattern.findall(suggestion_text)

    if matches:
        for _, title, content in matches:
            # Kod bloklarını tespit et
            code_blocks = code_pattern.findall(content)
            suggestions.append({
                'title': title.strip(),
                'content': content.strip(),
                'code': code_blocks if code_blocks else None  
            })
    else:
        paragraphs = [p.strip() for p in suggestion_text.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            if '**' in paragraph:
                title_match = re.search(r'\*\*([^*]+)\*\*\s*(.+)', paragraph, re.DOTALL)
                if title_match:
                    title, content = title_match.groups()
                    code_blocks = code_pattern.findall(content)
                    suggestions.append({
                        'title': title.strip(),
                        'content': '<pre class="code-block"><code>' + content.strip() + '</code></pre>',
                        'code': [code.replace('\n', '&#10;') for code in code_blocks] if code_blocks else None
                    })
                else:
                    suggestions.append({'title': '', 'content': paragraph})
            else:
                suggestions.append({'title': '', 'content': paragraph})

    return suggestions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']

        if not allowed_file(file.filename):
            return render_template('upload.html', error="You can upload just python file!")

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        code = encoding_detect(file_path)
        analysis, ai_suggestion, rating = analyze_with_ai(file_path, code)
        input_data = extract_metrics_from_file(file_path)
        if isinstance(input_data, str):
            return render_template('upload.html', error=input_data)

    elif 'github_url' in request.form and request.form['github_url'] != '':
        github_url = request.form['github_url']
        file_path = download_github_file(github_url)
        if not file_path:
            return render_template('upload.html', error="invalid github link!")
        code = encoding_detect(file_path)
        analysis, ai_suggestion, rating = analyze_with_ai(file_path, code)
        input_data = extract_metrics_from_file(file_path)
        if isinstance(input_data, str):
            return render_template('upload.html', error=input_data)

    else:
        return render_template('upload.html', error="Please upload a file or provide a valid GitHub URL!")

    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        gc.collect()

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    result = int(prediction[0])
    nolc = input_data['numberOfLinesOfCode'].iloc[0]
    noc = input_data['noc'].iloc[0]
    cbo = input_data['cbo'].iloc[0]
    dit = input_data['dit'].iloc[0]
    noa = input_data['numberOfAttributes'].iloc[0]
    lcom = input_data['lcom'].iloc[0]
    nom = input_data['numberOfMethods'].iloc[0]
    rfc = input_data['rfc'].iloc[0]
    wmc = input_data['wmc'].iloc[0]
    fanIn = input_data['fanIn'].iloc[0]
    fanOut = input_data['fanOut'].iloc[0]

    suggestions_list = parse_ai_suggestions(ai_suggestion)

    return render_template('index.html', result=result, noc=noc, cbo=cbo, dit=dit, noa=noa, lcom=lcom, nom=nom,
                           rfc=rfc, wmc=wmc, fanIn=fanIn, fanOut=fanOut, nolc=nolc, analysis=analysis,
                           ai_suggestion=suggestions_list, rating=rating)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
