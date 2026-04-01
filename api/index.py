import os
import time
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from PIL import Image

app = Flask(__name__, template_folder='../templates')

# المفتاح الذهبي الخاص بك
API_KEY = "AIzaSyCLvKWdyd44BEuPBEQuj0xkXi9hgLbGY9U"
genai.configure(api_key=API_KEY)

# استخدام الإصدار الأحدث لتجنب خطأ 404
model = genai.GenerativeModel('gemini-1.5-flash-latest')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.form.get('msg', '')
    file = request.files.get('file')
    mode = request.form.get('mode', 'ask')
    
    content_parts = []
    
    # تحديد وظيفة الـ AI بناءً على الاختيار
    instructions = {
        "summarize": "لخص هذا المحتوى في نقاط مركزة وواضحة.",
        "exam": "أنشئ اختباراً من 5 أسئلة اختيار من متعدد بناءً على هذا المحتوى.",
        "explain": "اشرح محتوى هذا الملف (فيديو/صورة/نص) بالتفصيل المبسط.",
        "ask": "أجب على السؤال التالي بناءً على الملف المرفق:"
    }
    
    system_prompt = instructions.get(mode, instructions["ask"])
    if user_msg:
        content_parts.append(f"{system_prompt}\n{user_msg}")
    else:
        content_parts.append(system_prompt)

    try:
        if file:
            mime = file.content_type
            # دعم الفيديو
            if 'video' in mime:
                temp_path = f"/tmp/{file.filename}"
                file.save(temp_path)
                g_file = genai.upload_file(path=temp_path)
                while g_file.state.name == "PROCESSING":
                    time.sleep(2)
                    g_file = genai.get_file(g_file.name)
                content_parts.append(g_file)
            # دعم الصور
            elif 'image' in mime:
                img = Image.open(file.stream)
                content_parts.append(img)
            # دعم الـ PDF
            elif 'pdf' in mime:
                reader = PdfReader(file)
                text = "سياق المنهج:\n" + "".join([p.extract_text() for p in reader.pages])
                content_parts.append(text[:15000])

        if not content_parts:
            return jsonify({"reply": "يا محمد، لم ترسل أي بيانات لمعالجتها!"})

        response = model.generate_content(content_parts)
        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"reply": f"خطأ في النظام الذكي: {str(e)}"})

app = app
