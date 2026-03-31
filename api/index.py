import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

app = Flask(__name__, template_folder='../templates')

# إعدادات الأمان والمجلدات لـ Vercel
UPLOAD_FOLDER = '/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# المفتاح الخاص بك (تم دمجه بالكامل)
API_KEY = "AIzaSyCLvKWdyd44BEuPBEQuj0xkXi9hgLbGY9U"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# قاعدة بيانات في الذاكرة (In-Memory) لتناسب Vercel
client = chromadb.Client()
collection = client.get_or_create_collection(name="mb_gold_vault")

def get_embedding(text):
    result = genai.embed_content(model="models/embedding-001", content=text, task_type="retrieval_document")
    return result['embedding']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم العثور على ملف"}), 400
    
    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    try:
        reader = PdfReader(path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                chunks = text_splitter.split_text(text)
                for j, chunk in enumerate(chunks):
                    all_chunks.append(chunk)
                    all_embeddings.append(get_embedding(chunk))
                    all_metadatas.append({"page": i + 1, "source": file.filename})
                    all_ids.append(f"{file.filename}_{i}_{j}")

        # إضافة كل البيانات دفعة واحدة للسرعة
        collection.add(
            embeddings=all_embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        return jsonify({"status": "success", "message": f"تم بنجاح! تم استيعاب {len(reader.pages)} صفحة."})
    except Exception as e:
        return jsonify({"error": f"فشل المعالجة: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('question', '')
    mode = data.get('mode', 'ask')

    if not query and mode == 'ask':
        return jsonify({"answer": "من فضلك اكتب سؤالك أولاً."})

    # البحث الدلالي المتطور
    search_query = query if query else "ملخص شامل للمحتوى"
    results = collection.query(query_embeddings=[get_embedding(search_query)], n_results=4)
    
    if not results['documents'] or not results['documents'][0]:
        return jsonify({"answer": "يرجى رفع ملف المنهج (PDF) أولاً ليتمكن الذكاء الاصطناعي من الإجابة."})

    context = "\n\n".join(results['documents'][0])
    sources = list(set([str(m['page']) for m in results['metadatas'][0]]))

    # توجيه الـ AI بناءً على الطلب
    prompts = {
        "quiz": f"بناءً على النص التالي: {context}\n ضع سؤالاً واحداً اختيار من متعدد مع 4 خيارات، ولا تذكر الإجابة الصحيحة إلا إذا طلبتها منك.",
        "summarize": f"بناءً على النص التالي: {context}\n قم بتلخيص أهم النقاط في شكل قائمة (bullet points) وبأسلوب مبسط.",
        "ask": f"أنت مساعد دراسي ذكي. أجب بدقة من النص المرفق فقط: {context}\n\nالسؤال: {query}"
    }

    try:
        response = model.generate_content(prompts.get(mode, prompts['ask']))
        return jsonify({
            "answer": response.text,
            "pages": " | ".join(sources)
        })
    except Exception as e:
        return jsonify({"answer": "عذراً، حدث خطأ في التواصل مع المحرك."})

# لضمان عمل Flask على Vercel
app = app
