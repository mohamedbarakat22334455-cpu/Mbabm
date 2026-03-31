import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from PIL import Image
import time

app = Flask(__name__, template_folder='../templates')

# المفتاح الذهبي
API_KEY = "AIzaSyCLvKWdyd44BEuPBEQuj0xkXi9hgLbGY9U"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.form.get('msg', '')
    file = request.files.get('file')
    mode = request.form.get('mode', 'ask')
    
    content_parts = [user_msg] if user_msg else ["اشرح لي هذا الملف"]

    try:
        if file:
            mime_type = file.content_type
            # معالجة الفيديو (نظام الرفع الذكي)
            if 'video' in mime_type:
                temp_path = f"/tmp/{file.filename}"
                file.save(temp_path)
                video_file = genai.upload_file(path=temp_path, display_name=file.filename)
                # الانتظار حتى معالجة الفيديو
                while video_file.state.name == "PROCESSING":
                    time.sleep(2)
                    video_file = genai.get_file(video_file.name)
                content_parts.append(video_file)
            
            # معالجة الصور
            elif 'image' in mime_type:
                img = Image.open(file.stream)
                content_parts.append(img)
            
            # معالجة الـ PDF
            elif 'pdf' in mime_type:
                reader = PdfReader(file)
                text = "".join([page.extract_text() for page in reader.pages])
                content_parts.append(f"سياق المنهج:\n{text[:20000]}")

        response = model.generate_content(content_parts)
        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"reply": f"خطأ في النظام الذكي: {str(e)}"})

app = app
