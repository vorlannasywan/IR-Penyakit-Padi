from flask import Flask, render_template, request
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer
import pdfplumber
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Queries and ground truth for evaluation
queries = {
    "bacterial leaf blight": [
        "bacterial leaf blight", "gejala bacterial leaf blight",
        "penyakit hawar daun bakteri","hawar daun padi", "serangan hawar daun", "penyebab bacterial leaf blight"
    ],
    "brown spot": [
        "brown spot", "bercak coklat", "bintik coklat pada daun",
        "penyebab brown spot", "pencegahan bercak coklat"
    ],
    "leaf smut": [
        "leaf smut", "penyakit patek pada padi", "daun berjamur", "penyebab leaf smut",
        "serangan leaf smut", "gejala daun hitam"
    ]
}

ground_truth = {
    "bacterial leaf blight": [
        "51.+SMN0248+-+Alvina+Walascha+-+Alvina+Walascha.pdf",
        "326-1-596-1-10-20130827.pdf",
        "233-Article Text-418-1-10-20210804.pdf",
        "2215206703-Master_Thesis.pdf",
        "6337-Article Text-13769-1-10-20190211.pdf"
    ],
    "brown spot": [
        "225312-identifikasi-leafblight-5e0651c8.pdf",
        "51.+SMN0248+-+Alvina+Walascha+-+Alvina+Walascha.pdf",
        "35547-75676621129-1-PB.pdf",
        "2215206703-Master_Thesis.pdf",
        "666-1460-1-PB.pdf"
    ],
    "leaf smut": [
        "51.+SMN0248+-+Alvina+Walascha+-+Alvina+Walascha.pdf",
        "35547-75676621129-1-PB.pdf",
        "225312-identifikasi-leafblight-5e0651c8.pdf",
        "6337-Article Text-13769-1-10-20190211.pdf",
        "233-Article Text-418-1-10-20210804.pdf"
    ]
}

# Preprocessing Function
def preprocess_text(text):
    # Case Folding
    text = text.lower()
    
    # Tokenizing
    tokens = word_tokenize(text)
    
    # Stopwords Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Rejoin tokens
    return " ".join(tokens)

# Fungsi untuk membaca dokumen PDF dari folder
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()  # Menambahkan teks dari setiap halaman
    return text

# Fungsi untuk membaca semua dokumen PDF dari folder
def read_pdf_documents(directory_path):
    documents = []
    
    # Iterasi melalui file di dalam folder
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):  # Misalnya, hanya file PDF yang diproses
            pdf_path = os.path.join(directory_path, filename)
            content = extract_text_from_pdf(pdf_path)
            documents.append({
                "title": filename,
                "snippet": content,
            })
    return documents

# Mengurutkan dokumen berdasarkan relevansi dengan VSM
def rank_documents(query, documents):
    # Preprocessing pada query dan dokumen
    preprocessed_query = preprocess_text(query)
    preprocessed_docs = [preprocess_text(doc['snippet']) for doc in documents]

    # Menggunakan TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([preprocessed_query] + preprocessed_docs)
    
    # Menghitung cosine similarity
    query_vector = vectors[0]
    doc_vectors = vectors[1:]
    similarities = (doc_vectors * query_vector.T).toarray()

    # Mengurutkan dokumen berdasarkan relevansi dan mengambil top 5
    ranked_docs = sorted(
        zip(similarities, documents),
        key=lambda x: x[0],
        reverse=True
    )[:5]  # Limit to top 5 documents
    
    # Mengembalikan dokumen yang sudah diurutkan
    return [doc for _, doc in ranked_docs]

# Ekstraksi informasi gejala, penyebab, pencegahan, pengertian
def extract_description_from_results(results):
    descriptions = {
        "gejala": [],
        "penyebab": [],
        "pencegahan": [],
        "pengertian": [],
    }

    # Ekstraksi dari hasil pencarian berdasarkan kategori
    for result in results:
        snippet = result["snippet"].lower()
        title = result["title"].lower()
        combined_text = f"{title} {snippet}"

        # Tokenizing dokumen menjadi kalimat
        sentences = sent_tokenize(result["snippet"])

        # Mencari kalimat yang mengandung kata kunci
        if re.search(r"(gejala|symptom|symptoms|tanda|manifestasi|indikasi|ciri-ciri|keluhan|tanda-tanda)", combined_text):
            for sentence in sentences:
                if re.search(r"(gejala|symptom|symptoms|tanda|manifestasi|indikasi|ciri-ciri|keluhan|tanda-tanda)", sentence):
                    descriptions["gejala"].append(sentence)

        if re.search(r"(penyebab|cause|etiology|faktor|pemicu|alasan|sumber|asal-usul)", combined_text):
            for sentence in sentences:
                if re.search(r"(penyebab|cause|etiology|faktor|pemicu|alasan|sumber|asal-usul)", sentence):
                    descriptions["penyebab"].append(sentence)

        if re.search(r"(pencegahan|prevention|mencegah|menghindari|antisipasi|langkah-langkah)", combined_text):
            for sentence in sentences:
                if re.search(r"(pencegahan|prevention|mencegah|menghindari|antisipasi|langkah-langkah)", sentence):
                    descriptions["pencegahan"].append(sentence)

        if re.search(r"(pengertian|definisi|arti|maksud|penjelasan|merupakan|disebut|diartikan)", combined_text):
            for sentence in sentences:
                if re.search(r"(pengertian|definisi|arti|maksud|penjelasan|merupakan|disebut|diartikan)", sentence):
                    descriptions["pengertian"].append(sentence)

    # Fallback jika tidak ditemukan
    for key in descriptions.keys():
        descriptions[key] = descriptions[key][:5]  # Limit to 5 points per category
        if not descriptions[key]:
            descriptions[key] = [result["snippet"] for result in results][:5]  # Limit to 5 snippets

    return descriptions

# Evaluasi akurasi, presisi, recall, dan f1 score berdasarkan ground truth
def evaluate_metrics(results, ground_truth):
    metrics = {}

    for disease, gt_docs in ground_truth.items():
        # Get document titles from the ranked results
        ranked_docs = [result["title"] for result in results]

        # Calculate True Positives (TP), False Positives (FP), False Negatives (FN)
        tp = len(set(gt_docs).intersection(set(ranked_docs)))  # Correct documents in results
        fp = len(set(ranked_docs) - set(gt_docs))  # Incorrect documents in results
        fn = len(set(gt_docs) - set(ranked_docs))  # Documents in ground truth but not in results
        
        # Calculate accuracy, precision, and recall
        accuracy = tp / len(gt_docs) if len(gt_docs) > 0 else 0  # Accuracy based on ground truth size
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision = TP / (TP + FP)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall = TP / (TP + FN)

        # Calculate F1 Score (harmonic mean of precision and recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[disease] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return metrics

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form['query']

        # Ambil dokumen PDF dari folder
        directory_path = 'static/documents/'  # Ganti dengan path ke folder dokumen PDF Anda
        documents = read_pdf_documents(directory_path)

        if documents:
            # Proses ranking dan ekstraksi informasi
            ranked_results = rank_documents(user_query, documents)
            descriptions = extract_description_from_results(ranked_results)

            # Evaluasi akurasi, presisi, recall, dan f1 score
            metrics = evaluate_metrics(ranked_results, ground_truth)
            
            return render_template('results.html', descriptions=descriptions, query=user_query, ranked_results=ranked_results, metrics=metrics)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)