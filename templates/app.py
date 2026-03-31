import os
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

app = Flask(__name__)
UPLOAD_FOLDER = 'my_private_docs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. إعداد Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. إعداد قاعدة البيانات (الذاكرة)
chroma_client = chromadb.Client()
# مسح الذاكرة القديمة وبدء واحدة جديدة لضمان عدم التداخل
try:
    chroma_client.delete_collection(name="ultimate_notebook")
except:
    pass
collection = chroma_client.create_collection(name="ultimate_notebook")

# 3. دالة التحويل الرقمي (لضمان دقة البحث بالعربي)
def get_embedding(text):
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return jsonify({"error": "لم يتم رفع ملف"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    reader = PdfReader(path)
    
    # مقسم النصوص (عشان لو الصفحة مليانة كلام الـ AI ميتوهش)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    total_chunks = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks = text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                # تخزين كل جزء مع رقم الصفحة بتاعته
                collection.add(
                    embeddings=[get_embedding(chunk)],
                    documents=[chunk],
                    metadatas=[{"page": i + 1, "source": file.filename}],
                    ids=[f"{file.filename}_p{i}_c{j}"]
                )
                total_chunks += 1

    return jsonify({"message": f"تم استيعاب المنهج! ({len(reader.pages)} صفحة)"})

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    mode = data.get('mode', 'ask')

    if not question and mode == 'ask':
        return jsonify({"answer": "اكتب سؤالك أولاً."}), 400

    # البحث الدقيق باستخدام الـ Embeddings
    query_text = question if question else "ملخص أهم النقاط"
    query_embedding = get_embedding(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3 # نجيب أدق 3 أجزاء
    )
    
    if not results['documents'][0]:
        return jsonify({"answer": "يرجى رفع المنهج أولاً."})

    context = "\n".join(results['documents'][0])
    sources = [f"ص {m['page']}" for m in results['metadatas'][0]]

    # التحكم في الـ Prompt حسب الميزة
    if mode == 'quiz':
        prompt = f"بناءً على هذا الجزء من المنهج: {context}\n ضع سؤالاً واحداً ذكياً لاختبار فهمي، ولا تكتب الإجابة."
    elif mode == 'summarize':
        prompt = f"بناءً على هذا الجزء من المنهج: {context}\n لخص أهم المعلومات في نقاط واضحة ومباشرة."
    else:
        prompt = f"أنت مساعد دراسي. أجب بدقة من المنهج المرفق فقط. إذا لم تجد المعلومة قل 'غير موجودة في المنهج'.\n\nالمنهج:\n{context}\n\nالسؤال: {question}"

    try:
        response = model.generate_content(prompt)
        # تنسيق الرد
        formatted_answer = response.text.replace('\n', '<br>')
        return jsonify({
            "answer": formatted_answer, 
            "sources": " | ".join(set(sources))
        })
    except Exception as e:
        return jsonify({"answer": f"خطأ تقني: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
