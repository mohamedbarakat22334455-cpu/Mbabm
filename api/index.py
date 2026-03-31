import os
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

app = Flask(__name__, template_folder='../templates')

# إعداد المفتاح والذكاء الاصطناعي
API_KEY = "AIzaSyCLvKWdyd44BEuPBEQuj0xkXi9hgLbGY9U"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# قاعدة بيانات في الذاكرة (سريعة جداً ومناسبة للـ Free Tier)
db_client = chromadb.Client()
vault = db_client.get_or_create_collection(name="gold_notebook_v4")

def get_embed(text):
    return genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")['embedding']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "ارفع ملف يا بطل"}), 400
    
    pdf = request.files['file']
    try:
        reader = PdfReader(pdf)
        text_full = ""
        for page in reader.pages:
            text_full += page.extract_text() + "\n"
        
        # تقسيم النص لضمان دقة الإجابة
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(text_full)
        
        # تخزين في الذاكرة الذهبية
        for idx, chunk in enumerate(chunks):
            vault.add(
                embeddings=[get_embed(chunk)],
                documents=[chunk],
                metadatas=[{"source": pdf.filename}],
                ids=[f"id_{idx}"]
            )
        return jsonify({"status": "Success", "msg": f"تم تجهيز {len(reader.pages)} صفحة بنجاح!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('msg', '')
    mode = request.json.get('mode', 'ask')
    
    # البحث عن المعلومات
    search = vault.query(query_embeddings=[get_embed(user_msg if user_msg else "ملخص")], n_results=3)
    context = "\n".join(search['documents'][0]) if search['documents'][0] else ""

    if not context:
        return jsonify({"reply": "ارفع المنهج الأول عشان أقدر أجاوبك بدقة."})

    # البرومبت الذهبي
    prompts = {
        "ask": f"أجب باحترافية من المنهج فقط: {context}\nالسؤال: {user_msg}",
        "summary": f"لخص المنهج التالي في نقاط ذهبية مركزة: {context}",
        "quiz": f"استخرج سؤالاً ذكياً من هذا النص: {context}"
    }

    response = model.generate_content(prompts.get(mode, prompts['ask']))
    return jsonify({"reply": response.text})

app = app
