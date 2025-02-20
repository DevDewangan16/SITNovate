from flask import Flask, render_template, request, send_file
import os
import requests
import json
from fpdf import FPDF

app = Flask(__name__)
AI_STUDIO_API_KEY = "AIzaSyBnCj6ntqFjrwDn0Cr-i3N20QMeIdxW3qA"  # Your API key

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

def analyze_text_with_ai_studio(text):
    """Analyze text using Google AI Studio API."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": AI_STUDIO_API_KEY}
    
    data = {
        "contents": [{"parts": [{"text": f"Analyze this text and provide insights: {text}"}]}]
    }
    
    response = requests.post(url, headers=headers, params=params, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No insights available")
    else:
        return f"Error: {response.status_code} - {response.text}"

def generate_pdf_report(original_text, analysis):
    """Generate a PDF report of the text analysis."""
    output_path = "static/reports/analysis_report.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Text Analysis Report", ln=True, align="C")
    pdf.ln(10)

    # Original Text
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Original Text:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, original_text[:1000] + "... (truncated)" if len(original_text) > 1000 else original_text)
    pdf.ln(5)

    # AI Analysis
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Google AI Studio Analysis:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 8, analysis)
    pdf.ln(5)

    # Save the PDF
    pdf.output(output_path)
    return output_path

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return render_template('index.html', error="No file selected")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No file selected")
    
    if not file.filename.endswith('.txt'):
        return render_template('index.html', error="Please upload a text file (.txt)")
    
    try:
        text_content = file.read().decode('utf-8')
        ai_analysis = analyze_text_with_ai_studio(text_content)
        pdf_path = generate_pdf_report(text_content, ai_analysis)
        return render_template('result.html', analysis_preview=ai_analysis[:500] + "..." if len(ai_analysis) > 500 else ai_analysis)
    except Exception as e:
        return render_template('index.html', error=f"Error processing file: {str(e)}")

@app.route('/download-report')
def download_report():
    return send_file("static/reports/analysis_report.pdf", as_attachment=True)

if __name__ == '__main__':
    os.makedirs('static/reports', exist_ok=True)
    app.run(debug=True,port=5001)