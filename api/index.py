import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from PIL import Image
import io

app = Flask(__name__, template_folder='../templates')

# المفتاح الذهبي بتاعك (تم دمجه بالكامل)
API_KEY = "AIzaSyCLvKWdyd44BEuPBEQuj0xkXi9hgLbGY9U"
genai.configure(api_key=API_KEY)

# إعداد الموديل (النسخة الخارقة اللي بتفهم صور ونصوص)
model = genai.GenerativeModel('gemini-1.5-flash')

# ذاكرة مؤقتة لجلسة الشات عشان يفتكر السياق
chat_session = model.start_chat(history=[])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.form.get('msg', '')
    file = request.files.get('file')
    # الميزة الجديدة: تحديد وضع الـ AI
    mode = request.form.get('mode', 'ask') 
    
    content_parts = []

    try:
        if file:
            filename = file.filename.lower()
            # لو الملف صورة (مسألة رياضية مثلاً)
            if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                img = Image.open(file.stream)
                content_parts.append(img)
            # لو الملف PDF (منهج كامل)
            elif filename.endswith('.pdf'):
                reader = PdfReader(file)
                pdf_text = "سياق المنهج من ملف PDF:\n"
                for page in reader.pages:
                    pdf_text += page.extract_text() + "\n"
                # نبعت أول 30 ألف حرف عشان Vercel Free Level
                content_parts.append(f"هذا هو محتوى المنهج، ذاكره جيداً لتجيبني منه لاحقاً: {pdf_text[:30000]}")
        
        # تنسيق البرومبت بناءً على الوضع المختار
        if mode == 'summary':
            prompt = f"من المنهج المرفق، لخص لي الأهم في نقاط ذهبية مركزة وبسيطة:\n {user_msg}"
        elif mode == 'quiz':
            prompt = f"من المنهج المرفق، اختبر ذكائي بسؤال واحد صعب اختيار من متعدد MCQ ولا تذكر الإجابة إلا إذا طلبتها منك:\n {user_msg}"
        else:
            prompt = f"أنت مساعد دراسي ذكي جداً، أجبني باحترافية وبأسلوب بسيط ومفصل من المنهج المرفق فقط:\n {user_msg}"

        content_parts.append(prompt)

        # إرسال البيانات لـ Gemini
        response = chat_session.send_message(content_parts)
        return jsonify({"reply": response.text})

    except Exception as e:
        return jsonify({"reply": f"حصلت مشكلة في الاتصال بذكائي، جرب تاني يا بطل. الخطأ: {str(e)}"})

app = app
