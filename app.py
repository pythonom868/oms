import streamlit as st
import os
import zipfile
import tempfile
import pandas as pd
import numpy as np
import pickle
import io
import time
import re
import json
import csv
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Optional imports with error handling
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    st.error("Tesseract OCR not available. Please install pytesseract, pdf2image, and Pillow.")

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not available. RNN models will be disabled.")

try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download NLTK resources safely
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        pass
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Some text processing features will be limited.")

try:
    import xlsxwriter
    XLSX_AVAILABLE = True
except ImportError:
    XLSX_AVAILABLE = False

# Directory paths for model persistence
MODEL_DIR = "models"
DATA_DIR = "training_data"
EXPORT_DIR = "exports"
SUMMARY_DIR = "summaries"

# Create directories if they don't exist
for directory in [MODEL_DIR, DATA_DIR, EXPORT_DIR, SUMMARY_DIR]:
    os.makedirs(directory, exist_ok=True)

# Paths for model files
NB_MODEL_PATH = os.path.join(MODEL_DIR, "nb_model.pkl")
RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.keras")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
SUMMARIZER_PATH = os.path.join(MODEL_DIR, "summarizer.pkl")

# ----------- Helper Functions -----------

def get_tesseract_path():
    """Handle Tesseract path based on platform"""
    if os.name == 'nt':  # Windows
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\tesseract\tesseract.exe'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    return 'tesseract'  # Default to system PATH

# Try to set Tesseract path if available
if TESSERACT_AVAILABLE:
    try:
        pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()
    except Exception as e:
        st.warning(f"Could not set Tesseract path: {e}")

# ----------- Error Handling -----------

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

def safe_process(func):
    """Decorator for safer processing with more detailed error messages"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error in {func.__name__}: {str(e)}"
            st.error(error_msg)
            return None
    return wrapper

# ----------- ML Functions -----------

def train_naive_bayes(X, y):
    """Train a Naive Bayes classifier"""
    try:
        model = make_pipeline(CountVectorizer(max_features=5000), MultinomialNB())
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error training Naive Bayes: {e}")
        return None

def create_rnn_model(input_length, num_classes, embedding_dim=128):
    """Create an RNN model with improved architecture"""
    if not TF_AVAILABLE:
        return None
    
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=10000, output_dim=embedding_dim, input_length=input_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error creating RNN model: {e}")
        return None

# ----------- Text Extraction -----------

@safe_process
def extract_text_from_image(image, lang='eng'):
    """Extract text from an image with language support"""
    if not TESSERACT_AVAILABLE:
        st.error("Tesseract OCR not available")
        return ""
    
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

@safe_process
def extract_text_from_pdf(pdf_file, lang='eng'):
    """Extract text from PDF with language support"""
    if not TESSERACT_AVAILABLE:
        st.error("PDF processing not available")
        return ""
    
    try:
        images = convert_from_path(pdf_file)
        texts = []
        for image in images:
            text = extract_text_from_image(image, lang=lang)
            if text:
                texts.append(text)
        return "\n".join(texts)
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return ""

# ----------- Text Analysis -----------

def preprocess_text(text):
    """Preprocess text for NLP tasks"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text, top_n=10):
    """Extract keywords from text using TF-IDF"""
    try:
        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
        # If text is too short, return empty list
        if len(text.split()) < 5:
            return []
        
        # Fit transform on the text
        response = tfidf.fit_transform([text])
        
        # Get feature names and scores
        feature_names = tfidf.get_feature_names_out()
        scores = response.toarray()[0]
        
        # Sort by scores
        keywords = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        keywords.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N keywords
        return keywords[:top_n]
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return []

def simple_sentence_split(text):
    """Simple sentence splitting fallback"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_summary(text, max_sentences=5):
    """Generate a summary using extractive summarization"""
    if not text or len(text) < 100:
        return text
    
    try:
        # Split text into sentences
        if NLTK_AVAILABLE:
            try:
                sentences = sent_tokenize(text)
            except:
                sentences = simple_sentence_split(text)
        else:
            sentences = simple_sentence_split(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Clean sentences
        clean_sentences = [preprocess_text(sentence) for sentence in sentences]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        sentence_vectors = vectorizer.fit_transform(clean_sentences)
        
        # Calculate sentence scores based on TF-IDF sum
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = np.sum(sentence_vectors[i].toarray())
            sentence_scores.append((i, score))
        
        # Sort sentences by score and take top N
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in sentence_scores[:max_sentences]]
        
        # Sort indices to maintain original order
        top_indices.sort()
        
        # Create summary
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return text[:500] + "..." if len(text) > 500 else text

def compare_documents(text1, text2):
    """Compare two documents and find similarities"""
    # Clean texts
    clean_text1 = preprocess_text(text1)
    clean_text2 = preprocess_text(text2)
    
    # Get similarity ratio
    similarity = difflib.SequenceMatcher(None, clean_text1, clean_text2).ratio()
    
    # Use TF-IDF for content comparison
    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform([clean_text1, clean_text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0
    
    # Find common phrases
    tokens1 = set(clean_text1.split())
    tokens2 = set(clean_text2.split())
    common_words = tokens1.intersection(tokens2)
    
    return {
        "similarity_ratio": similarity,
        "cosine_similarity": cosine_sim,
        "common_word_count": len(common_words),
        "common_words": list(common_words)[:20]  # Limit to 20 words
    }

# ----------- File Processing -----------

def process_files(file_list, language='eng'):
    """Process files and extract text with language support"""
    dataset = []
    all_text = ""
    classification_counts = {}
    individual_texts = {}
    summaries = {}
    keywords = {}
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        label = os.path.splitext(filename)[0]
        
        try:
            text = ""
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(filepath, lang=language)
            else:
                if TESSERACT_AVAILABLE:
                    text = extract_text_from_image(Image.open(filepath), lang=language)
                else:
                    st.warning(f"Cannot process {filename}: OCR not available")
                    continue
            
            # Generate summary and keywords if text is extracted
            if text and text.strip():
                summary = generate_summary(text)
                doc_keywords = extract_keywords(text)
                
                classification_counts[label] = classification_counts.get(label, 0) + 1
                dataset.append({
                    "filename": filename,
                    "text": text.strip(),
                    "label": label,
                    "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "language": language,
                    "char_count": len(text),
                    "word_count": len(text.split())
                })
                
                individual_texts[filename] = text.strip()
                summaries[filename] = summary
                keywords[filename] = doc_keywords
                all_text += f"\n\n--- {filename} ---\n{text.strip()[:500]}...\n"
            else:
                st.warning(f"⚠️ No text extracted from {filename}")
        except Exception as e:
            st.error(f"❌ Failed to process {filename}: {e}")
    
    return dataset, all_text, classification_counts, individual_texts, summaries, keywords

def get_files_from_upload(uploaded_files):
    """Process uploaded files"""
    file_list = []
    for file in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())
        file_list.append(temp_path)
    return file_list

def get_files_from_zip(uploaded_zip):
    """Extract files from a zip archive"""
    file_list = []
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, uploaded_zip.name)
        
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
            
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp')):
                    file_list.append(os.path.join(root, file))
    return file_list

def get_files_from_folder(folder_path):
    """Get files from a local folder"""
    file_list = []
    if os.path.exists(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp')):
                    file_list.append(os.path.join(root, file))
    return file_list

# ----------- Model Management -----------

def save_model(model, filename):
    """Save model to file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False
    
def load_model(filename):
    """Load model from file with error handling"""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model {filename}: {e}")
    return None

def save_training_data(df):
    """Save training data and merge with existing"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            existing_df = pd.read_csv(TRAINING_DATA_PATH)
            combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['filename'])
            combined_df.to_csv(TRAINING_DATA_PATH, index=False)
        else:
            df.to_csv(TRAINING_DATA_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving training data: {e}")
        return False

def load_training_data():
    """Load saved training data if exists"""
    try:
        if os.path.exists(TRAINING_DATA_PATH):
            return pd.read_csv(TRAINING_DATA_PATH)
    except Exception as e:
        st.error(f"Error loading training data: {e}")
    return pd.DataFrame(columns=["filename", "text", "label", "processed_date"])

# ----------- Model Training and Prediction -----------

def train_models(df, auto_save=True):
    """Train models with the provided dataframe"""
    if len(df) < 2 or len(df["label"].unique()) < 2:
        st.warning("⚠️ Not enough data or unique labels to train model")
        return None, None, None, None, None
    
    unique_labels = df["label"].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = df["label"].map(label_map)
    texts = df["text"].fillna("").tolist()
    
    with st.spinner("Training Naive Bayes model..."):
        nb_model = train_naive_bayes(texts, y)
        if nb_model is None:
            return None, None, None, None, None
    
    # RNN model (only if TensorFlow is available)
    rnn_model = None
    tokenizer = None
    max_length = 100
    
    if TF_AVAILABLE:
        with st.spinner("Preparing RNN data..."):
            try:
                tokenizer = Tokenizer(num_words=10000)
                tokenizer.fit_on_texts(texts)
                sequences = tokenizer.texts_to_sequences(texts)
                max_length = min(max([len(seq) for seq in sequences]) if sequences else 100, 500)
                padded_seq = pad_sequences(sequences, maxlen=max_length)
            except Exception as e:
                st.error(f"Error preparing RNN data: {e}")
                tokenizer = None
        
        if tokenizer is not None:
            with st.spinner("Training RNN model..."):
                try:
                    rnn_model = create_rnn_model(max_length, len(unique_labels))
                    if rnn_model is not None:
                        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='loss', patience=3, restore_best_weights=True
                        )
                        rnn_model.fit(
                            padded_seq, 
                            y, 
                            epochs=5, 
                            batch_size=32,
                            callbacks=[early_stopping],
                            verbose=0
                        )
                except Exception as e:
                    st.error(f"Error training RNN model: {e}")
                    rnn_model = None
    
    # Save the models if auto_save is True
    if auto_save:
        save_model(nb_model, NB_MODEL_PATH)
        
        if rnn_model:
            try:
                rnn_model.save(RNN_MODEL_PATH)
            except Exception as e:
                st.warning(f"Could not save RNN model: {e}")
        
        metadata = {
            'tokenizer': tokenizer,
            'label_map': label_map,
            'unique_labels': unique_labels,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'num_samples': len(df),
            'max_sequence_length': max_length
        }
        save_model(metadata, METADATA_PATH)
    
    return nb_model, rnn_model, tokenizer, label_map, unique_labels

def predict_with_models(df, nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata=None):
    """Make predictions using trained models with improved error handling"""
    texts = df["text"].fillna("").tolist()
    result_df = df.copy()
    
    max_length = 100
    if metadata and 'max_sequence_length' in metadata:
        max_length = metadata['max_sequence_length']
    
    # Naive Bayes prediction
    try:
        nb_preds = nb_model.predict(texts)
        nb_labels = [unique_labels[pred] for pred in nb_preds]
        result_df["NaiveBayes_Label"] = nb_labels
    except Exception as e:
        st.error(f"Error in Naive Bayes prediction: {e}")
        result_df["NaiveBayes_Label"] = ["Error" for _ in range(len(texts))]
    
    # RNN prediction
    if rnn_model and tokenizer:
        try:
            sequences = tokenizer.texts_to_sequences(texts)
            padded_seq = pad_sequences(sequences, maxlen=max_length)
            rnn_preds = rnn_model.predict(padded_seq)
            rnn_labels = [unique_labels[np.argmax(pred)] for pred in rnn_preds]
            
            confidence_scores = np.max(rnn_preds, axis=1)
            
            result_df["RNN_Label"] = rnn_labels
            result_df["RNN_Confidence"] = [f"{score:.2%}" for score in confidence_scores]
            
            result_df["Prediction"] = [
                f"NB: {nb} | RNN: {rnn} ({conf})"
                for nb, rnn, conf in zip(
                    result_df["NaiveBayes_Label"], 
                    rnn_labels, 
                    [f"{score:.2%}" for score in confidence_scores]
                )
            ]
        except Exception as e:
            st.error(f"Error in RNN prediction: {e}")
            result_df["RNN_Label"] = ["Error" for _ in range(len(texts))]
            result_df["RNN_Confidence"] = ["N/A" for _ in range(len(texts))]
            result_df["Prediction"] = [
                f"NB: {nb} | RNN: Error"
                for nb in result_df["NaiveBayes_Label"]
            ]
    else:
        result_df["RNN_Label"] = ["N/A" for _ in range(len(texts))]
        result_df["RNN_Confidence"] = ["N/A" for _ in range(len(texts))]
        result_df["Prediction"] = [f"NB: {nb} | RNN: N/A" for nb in result_df["NaiveBayes_Label"]]
    
    return result_df

def load_saved_models():
    """Load all saved models if they exist"""
    nb_model = load_model(NB_MODEL_PATH)
    metadata = load_model(METADATA_PATH)
    rnn_model = None
    
    if TF_AVAILABLE and os.path.exists(RNN_MODEL_PATH):
        try:
            rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading RNN model: {e}")
    
    if nb_model and metadata:
        return nb_model, rnn_model, metadata.get('tokenizer'), metadata.get('label_map'), metadata.get('unique_labels'), metadata
    
    return None, None, None, None, None, None

# ----------- Document Search Function -----------

def search_documents(documents, query, threshold=0.2):
    """Search through documents for a specific query"""
    if not query.strip() or not documents:
        return []
    
    results = []
    query = query.lower()
    
    for filename, text in documents.items():
        if not text:
            continue
            
        if query in text.lower():
            frequency = text.lower().count(query)
            position = text.lower().find(query) / len(text) if len(text) > 0 else 1
            score = (frequency * 0.7) + ((1 - position) * 0.3)
            
            idx = text.lower().find(query)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 100)
            context = text[start:end]
            
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
                
            results.append({
                "filename": filename,
                "score": min(score, 1.0),
                "context": context,
                "match_type": "direct"
            })
        else:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([query, text.lower()])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
                if similarity > threshold:
                    results.append({
                        "filename": filename,
                        "score": similarity,
                        "context": text[:200] + "..." if len(text) > 200 else text,
                        "match_type": "semantic"
                    })
            except Exception:
                continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ----------- K-Means Clustering Functions -----------

def get_optimal_clusters(vectorized_data, max_k=10):
    """Find optimal number of clusters using Elbow method"""
    n_samples = vectorized_data.shape[0]
    max_k = min(max_k, n_samples - 1)
    
    wcss = []
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectorized_data)
        wcss.append(kmeans.inertia_)
    
    # Find elbow point
    optimal_k = min(3, max_k)
    if KNEED_AVAILABLE:
        try:
            kl = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
            optimal_k = kl.elbow if kl.elbow else min(3, max_k)
        except Exception as e:
            st.warning(f"Error finding optimal clusters: {e}")
    
    return optimal_k, wcss

def create_vectorizer(df):
    """Create and fit TF-IDF vectorizer on document text"""
    vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words='english',
        min_df=1,
        max_df=0.9
    )
    vectorized_data = vectorizer.fit_transform(df["text"].fillna(""))
    save_model(vectorizer, VECTORIZER_PATH)
    return vectorizer, vectorized_data

def train_kmeans(vectorized_data, n_clusters=3):
    """Train K-means clustering model"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(vectorized_data)
    save_model(kmeans, KMEANS_MODEL_PATH)
    return kmeans

def reduce_dimensions(vectorized_data, method='pca', n_components=2, random_state=42):
    """Reduce dimensions of vectorized data for visualization"""
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data.toarray())
    elif method == 'svd':
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                      perplexity=min(30, vectorized_data.shape[0]-1))
        return reducer.fit_transform(vectorized_data.toarray())
    else:
        # Fallback to PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data.toarray())

def plot_elbow_curve(wcss):
    """Plot elbow curve to help find optimal K"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(wcss) + 1), wcss, 'bo-')
    ax.set_title('Elbow Method For Optimal k')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS (Within-Cluster Sum of Square)')
    ax.grid(True)
    return fig

def plot_clusters_2d(df, reduced_data, kmeans):
    """Plot 2D visualization of clusters"""
    df_plot = df.copy()
    df_plot['Cluster'] = kmeans.labels_
    
    fig = px.scatter(
        x=reduced_data[:, 0], 
        y=reduced_data[:, 1],
        color=kmeans.labels_,
        hover_name=df_plot['filename'],
        title="Document Clusters (2D View)",
        labels={'color': 'Cluster'},
        color_continuous_scale=px.colors.qualitative.Set1
    )
    
    return fig

def plot_clusters_3d(df, reduced_data_3d, kmeans):
    """Plot 3D visualization of clusters"""
    df_plot = df.copy()
    df_plot['Cluster'] = kmeans.labels_
    
    fig = px.scatter_3d(
        x=reduced_data_3d[:, 0], 
        y=reduced_data_3d[:, 1], 
        z=reduced_data_3d[:, 2],
        color=kmeans.labels_,
        hover_name=df_plot['filename'],
        title="Document Clusters (3D View)",
        labels={'color': 'Cluster'},
        color_continuous_scale=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )
    
    return fig

def get_cluster_summary(df, kmeans, vectorizer):
    """Get summary of each cluster"""
    cluster_summaries = {}
    
    for cluster_id in range(kmeans.n_clusters):
        cluster_docs = df[df['Cluster'] == cluster_id]
        
        if len(cluster_docs) > 0:
            # Get cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            
            # Get top terms for this cluster
            feature_names = vectorizer.get_feature_names_out()
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            
            cluster_summaries[cluster_id] = {
                'documents': cluster_docs['filename'].tolist(),
                'count': len(cluster_docs),
                'top_terms': top_terms,
                'avg_word_count': cluster_docs['word_count'].mean() if 'word_count' in cluster_docs.columns else 0
            }
    
    return cluster_summaries

# ----------- Export Functions -----------

def export_to_csv(df, filename="export.csv"):
    """Export dataframe to CSV"""
    try:
        csv_path = os.path.join(EXPORT_DIR, filename)
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        st.error(f"Error exporting to CSV: {e}")
        return None

def export_to_excel(df, filename="export.xlsx"):
    """Export dataframe to Excel"""
    if not XLSX_AVAILABLE:
        st.error("Excel export not available. Install xlsxwriter.")
        return None
    
    try:
        excel_path = os.path.join(EXPORT_DIR, filename)
        df.to_excel(excel_path, index=False, engine='xlsxwriter')
        return excel_path
    except Exception as e:
        st.error(f"Error exporting to Excel: {e}")
        return None

def export_summaries(summaries, filename="summaries.json"):
    """Export summaries to JSON"""
    try:
        json_path = os.path.join(SUMMARY_DIR, filename)
        with open(json_path, 'w') as f:
            json.dump(summaries, f, indent=2)
        return json_path
    except Exception as e:
        st.error(f"Error exporting summaries: {e}")
        return None

# ----------- Main Streamlit App -----------

def main():
    st.set_page_config(
        page_title="Advanced Document Processing & ML System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Advanced Document Processing & ML System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home",
        "📤 Upload & Process",
        "🤖 Train Models",
        "🔍 Predict & Classify",
        "📊 Document Analysis",
        "🔍 Search Documents",
        "📈 Clustering Analysis",
        "⚙️ Model Management",
        "📥 Export Data"
    ])
    
    # Language selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("OCR Settings")
    language = st.sidebar.selectbox(
        "OCR Language:",
        ["eng", "spa", "fra", "deu", "chi_sim", "chi_tra", "jpn", "kor", "rus", "ara"],
        help="Select the language for OCR text extraction"
    )
    
    # Initialize session state
    if 'dataset' not in st.session_state:
        st.session_state.dataset = []
    if 'all_text' not in st.session_state:
        st.session_state.all_text = ""
    if 'individual_texts' not in st.session_state:
        st.session_state.individual_texts = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'keywords' not in st.session_state:
        st.session_state.keywords = {}
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "📤 Upload & Process":
        upload_process_page(language)
    elif page == "🤖 Train Models":
        train_models_page()
    elif page == "🔍 Predict & Classify":
        predict_classify_page(language)
    elif page == "📊 Document Analysis":
        document_analysis_page()
    elif page == "🔍 Search Documents":
        search_documents_page()
    elif page == "📈 Clustering Analysis":
        clustering_analysis_page()
    elif page == "⚙️ Model Management":
        model_management_page()
    elif page == "📥 Export Data":
        export_data_page()

def home_page():
    """Home page with system overview"""
    st.header("🏠 Welcome to Advanced Document Processing System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Features")
        st.markdown("""
        - **Multi-format Support**: PDF, Images (PNG, JPG, TIFF, etc.)
        - **OCR Text Extraction**: Multiple language support
        - **Machine Learning**: Naive Bayes & RNN models
        - **Document Clustering**: K-means clustering with visualization
        - **Text Analysis**: Keyword extraction, summarization
        - **Document Search**: Semantic and direct text search
        - **Export Options**: CSV, Excel, JSON formats
        """)
    
    with col2:
        st.subheader("📊 System Status")
        
        # Check available libraries
        status_items = [
            ("Tesseract OCR", TESSERACT_AVAILABLE),
            ("TensorFlow", TF_AVAILABLE),
            ("NLTK", NLTK_AVAILABLE),
            ("Excel Export", XLSX_AVAILABLE),
            ("Knee Locator", KNEED_AVAILABLE)
        ]
        
        for item, available in status_items:
            status = "✅ Available" if available else "❌ Not Available"
            st.write(f"**{item}**: {status}")
    
    st.markdown("---")
    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", len(st.session_state.dataset))
    
    with col2:
        training_data = load_training_data()
        st.metric("Training Samples", len(training_data))
    
    with col3:
        model_exists = os.path.exists(NB_MODEL_PATH)
        st.metric("Models Trained", "Yes" if model_exists else "No")
    
    with col4:
        st.metric("Text Extracted", f"{len(st.session_state.all_text)} chars")

def upload_process_page(language):
    """Upload and process documents page"""
    st.header("📤 Upload & Process Documents")
    
    # Upload options
    upload_method = st.radio(
        "Choose upload method:",
        ["Upload Files", "Upload ZIP Archive", "Local Folder Path"]
    )
    
    file_list = []
    
    if upload_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            accept_multiple_files=True
        )
        if uploaded_files:
            file_list = get_files_from_upload(uploaded_files)
    
    elif upload_method == "Upload ZIP Archive":
        uploaded_zip = st.file_uploader(
            "Choose ZIP file:",
            type=['zip']
        )
        if uploaded_zip:
            file_list = get_files_from_zip(uploaded_zip)
    
    elif upload_method == "Local Folder Path":
        folder_path = st.text_input("Enter folder path:")
        if folder_path and st.button("Load from Folder"):
            file_list = get_files_from_folder(folder_path)
    
    if file_list:
        st.success(f"Found {len(file_list)} files to process")
        
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                dataset, all_text, classification_counts, individual_texts, summaries, keywords = process_files(file_list, language)
                
                # Update session state
                st.session_state.dataset = dataset
                st.session_state.all_text = all_text
                st.session_state.individual_texts = individual_texts
                st.session_state.summaries = summaries
                st.session_state.keywords = keywords
                
                st.success(f"✅ Processed {len(dataset)} documents successfully!")
                
                # Display results
                if dataset:
                    df = pd.DataFrame(dataset)
                    st.subheader("📊 Processing Results")
                    st.dataframe(df)
                    
                    # Classification counts
                    st.subheader("📋 Document Categories")
                    for label, count in classification_counts.items():
                        st.write(f"**{label}**: {count} documents")

def train_models_page():
    """Train machine learning models page"""
    st.header("🤖 Train Machine Learning Models")
    
    if not st.session_state.dataset:
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    df = pd.DataFrame(st.session_state.dataset)
    
    st.subheader("📊 Training Data Overview")
    st.write(f"Total documents: {len(df)}")
    st.write(f"Unique labels: {len(df['label'].unique())}")
    
    # Display label distribution
    label_counts = df['label'].value_counts()
    fig = px.bar(x=label_counts.index, y=label_counts.values, title="Label Distribution")
    st.plotly_chart(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_save = st.checkbox("Auto-save trained models", value=True)
    
    with col2:
        save_training_data_flag = st.checkbox("Save training data", value=True)
    
    if st.button("🚀 Train Models", type="primary"):
        if len(df) < 2 or len(df["label"].unique()) < 2:
            st.error("❌ Need at least 2 documents with different labels to train models")
            return
        
        # Save training data if requested
        if save_training_data_flag:
            save_training_data(df)
        
        # Train models
        nb_model, rnn_model, tokenizer, label_map, unique_labels = train_models(df, auto_save)
        
        if nb_model:
            st.success("✅ Naive Bayes model trained successfully!")
            st.session_state.models_trained = True
        
        if rnn_model:
            st.success("✅ RNN model trained successfully!")
        elif TF_AVAILABLE:
            st.warning("⚠️ RNN model training failed")
        else:
            st.info("ℹ️ RNN training skipped (TensorFlow not available)")

def predict_classify_page(language):
    """Predict and classify new documents page"""
    st.header("🔍 Predict & Classify Documents")
    
    # Load saved models
    nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata = load_saved_models()
    
    if not nb_model:
        st.warning("⚠️ No trained models found. Please train models first.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # Display model info
    if metadata:
        st.subheader("📊 Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", metadata.get('num_samples', 'N/A'))
        with col2:
            st.metric("Unique Labels", len(unique_labels) if unique_labels else 'N/A')
        with col3:
            st.metric("Last Updated", metadata.get('last_updated', 'N/A'))
    
    # Upload new documents for prediction
    st.subheader("📤 Upload Documents for Classification")
    
    uploaded_files = st.file_uploader(
        "Choose files to classify:",
        type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        file_list = get_files_from_upload(uploaded_files)
        
        if st.button("🔍 Classify Documents", type="primary"):
            with st.spinner("Processing and classifying documents..."):
                dataset, _, _, _, _, _ = process_files(file_list, language)
                
                if dataset:
                    df = pd.DataFrame(dataset)
                    
                    # Make predictions
                    result_df = predict_with_models(
                        df, nb_model, rnn_model, tokenizer, 
                        label_map, unique_labels, metadata
                    )
                    
                    st.subheader("📊 Classification Results")
                    st.dataframe(result_df[['filename', 'Prediction', 'word_count', 'char_count']])
                    
                    # Show detailed predictions
                    if st.checkbox("Show detailed predictions"):
                        st.dataframe(result_df)

def document_analysis_page():
    """Document analysis and comparison page"""
    st.header("📊 Document Analysis")
    
    if not st.session_state.individual_texts:
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    # Document summaries
    st.subheader("📝 Document Summaries")
    if st.session_state.summaries:
        selected_doc = st.selectbox("Select document for summary:", list(st.session_state.summaries.keys()))
        if selected_doc:
            st.write("**Summary:**")
            st.write(st.session_state.summaries[selected_doc])
    
    # Keywords extraction
    st.subheader("🔑 Keywords Analysis")
    if st.session_state.keywords:
        selected_doc_kw = st.selectbox("Select document for keywords:", list(st.session_state.keywords.keys()))
        if selected_doc_kw and st.session_state.keywords[selected_doc_kw]:
            keywords_df = pd.DataFrame(st.session_state.keywords[selected_doc_kw], columns=['Keyword', 'Score'])
            st.dataframe(keywords_df)
            
            # Keywords visualization
            fig = px.bar(keywords_df.head(10), x='Score', y='Keyword', orientation='h', 
                        title=f"Top Keywords in {selected_doc_kw}")
            st.plotly_chart(fig)
    
    # Document comparison
    st.subheader("🔀 Document Comparison")
    if len(st.session_state.individual_texts) >= 2:
        doc_names = list(st.session_state.individual_texts.keys())
        
        col1, col2 = st.columns(2)
        with col1:
            doc1 = st.selectbox("First document:", doc_names)
        with col2:
            doc2 = st.selectbox("Second document:", doc_names, index=1)
        
        if st.button("Compare Documents"):
            comparison = compare_documents(
                st.session_state.individual_texts[doc1],
                st.session_state.individual_texts[doc2]
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Similarity Ratio", f"{comparison['similarity_ratio']:.2%}")
            with col2:
                st.metric("Cosine Similarity", f"{comparison['cosine_similarity']:.2%}")
            with col3:
                st.metric("Common Words", comparison['common_word_count'])
            
            if comparison['common_words']:
                st.write("**Common Words:**")
                st.write(", ".join(comparison['common_words']))

def search_documents_page():
    """Search through processed documents"""
    st.header("🔍 Search Documents")
    
    if not st.session_state.individual_texts:
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    st.subheader("🔍 Search Query")
    query = st.text_input("Enter search query:")
    
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Similarity threshold:", 0.0, 1.0, 0.2, 0.05)
    with col2:
        max_results = st.number_input("Max results:", 1, 50, 10)
    
    if query and st.button("🔍 Search"):
        results = search_documents(st.session_state.individual_texts, query, threshold)
        
        if results:
            st.subheader(f"📊 Search Results ({len(results)} found)")
            
            for i, result in enumerate(results[:max_results]):
                with st.expander(f"#{i+1}: {result['filename']} (Score: {result['score']:.2%})"):
                    st.write(f"**Match Type:** {result['match_type'].title()}")
                    st.write(f"**Context:**")
                    st.write(result['context'])
        else:
            st.info("No matching documents found.")

def clustering_analysis_page():
    """Document clustering analysis page"""
    st.header("📈 Clustering Analysis")
    
    if not st.session_state.dataset:
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    df = pd.DataFrame(st.session_state.dataset)
    
    if len(df) < 3:
        st.warning("⚠️ Need at least 3 documents for clustering analysis.")
        return
    
    st.subheader("⚙️ Clustering Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        auto_clusters = st.checkbox("Auto-detect optimal clusters", value=True)
        if not auto_clusters:
            n_clusters = st.number_input("Number of clusters:", 2, min(10, len(df)-1), 3)
    
    with col2:
        reduction_method = st.selectbox("Dimension reduction method:", ['pca', 'svd', 'tsne'])
    
    if st.button("🚀 Perform Clustering Analysis", type="primary"):
        with st.spinner("Performing clustering analysis..."):
            # Create vectorizer and vectorized data
            vectorizer, vectorized_data = create_vectorizer(df)
            
            # Find optimal clusters if auto mode
            if auto_clusters:
                optimal_k, wcss = get_optimal_clusters(vectorized_data)
                n_clusters = optimal_k
                
                st.subheader("📊 Elbow Curve")
                fig_elbow = plot_elbow_curve(wcss)
                st.pyplot(fig_elbow)
                plt.close()
            
            # Train K-means
            kmeans = train_kmeans(vectorized_data, n_clusters)
            
            # Add cluster labels to dataframe
            df_clustered = df.copy()
            df_clustered['Cluster'] = kmeans.labels_
            
            # Dimension reduction for visualization
            reduced_2d = reduce_dimensions(vectorized_data, reduction_method, 2)
            reduced_3d = reduce_dimensions(vectorized_data, reduction_method, 3)
            
            # 2D Cluster visualization
            st.subheader("📊 2D Cluster Visualization")
            fig_2d = plot_clusters_2d(df_clustered, reduced_2d, kmeans)
            st.plotly_chart(fig_2d)
            
            # 3D Cluster visualization
            st.subheader("📊 3D Cluster Visualization")
            fig_3d = plot_clusters_3d(df_clustered, reduced_3d, kmeans)
            st.plotly_chart(fig_3d)
            
            # Cluster summaries
            st.subheader("📋 Cluster Summaries")
            cluster_summaries = get_cluster_summary(df_clustered, kmeans, vectorizer)
            
            for cluster_id, summary in cluster_summaries.items():
                with st.expander(f"Cluster {cluster_id} ({summary['count']} documents)"):
                    st.write(f"**Documents:** {', '.join(summary['documents'])}")
                    st.write(f"**Top Terms:** {', '.join(summary['top_terms'])}")
                    st.write(f"**Avg Word Count:** {summary['avg_word_count']:.1f}")

def model_management_page():
    """Model management page"""
    st.header("⚙️ Model Management")
    
    # Model status
    st.subheader("📊 Model Status")
    
    models_info = [
        ("Naive Bayes", NB_MODEL_PATH),
        ("RNN Model", RNN_MODEL_PATH),
        ("K-means", KMEANS_MODEL_PATH),
        ("Vectorizer", VECTORIZER_PATH),
        ("Metadata", METADATA_PATH)
    ]
    
    for model_name, model_path in models_info:
        exists = os.path.exists(model_path)
        size = os.path.getsize(model_path) if exists else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**{model_name}**")
        with col2:
            st.write("✅ Available" if exists else "❌ Not Found")
        with col3:
            st.write(f"{size/1024:.1f} KB" if exists else "0 KB")
    
    # Training data status
    st.subheader("📊 Training Data")
    training_df = load_training_data()
    if not training_df.empty:
        st.write(f"**Records:** {len(training_df)}")
        st.write(f"**Labels:** {', '.join(training_df['label'].unique())}")
        
        if st.checkbox("Show training data"):
            st.dataframe(training_df)
    else:
        st.write("No training data found.")
    
    # Model actions
    st.subheader("🔧 Model Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Models", type="secondary"):
            if st.checkbox("Confirm deletion"):
                for _, model_path in models_info:
                    if os.path.exists(model_path):
                        os.remove(model_path)
                st.success("All models cleared!")
    
    with col2:
        if st.button("🗑️ Clear Training Data", type="secondary"):
            if st.checkbox("Confirm training data deletion"):
                if os.path.exists(TRAINING_DATA_PATH):
                    os.remove(TRAINING_DATA_PATH)
                st.success("Training data cleared!")

def export_data_page():
    """Export data page"""
    st.header("📥 Export Data")
    
    if not st.session_state.dataset:
        st.warning("⚠️ No processed documents found. Please upload and process documents first.")
        return
    
    df = pd.DataFrame(st.session_state.dataset)
    
    st.subheader("📊 Data Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Document Data**")
        if st.button("📄 Export to CSV"):
            csv_path = export_to_csv(df, f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            if csv_path:
                st.success(f"Exported to: {csv_path}")
        
        if XLSX_AVAILABLE and st.button("📊 Export to Excel"):
            excel_path = export_to_excel(df, f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            if excel_path:
                st.success(f"Exported to: {excel_path}")
    
    with col2:
        st.write("**Summaries & Keywords**")
        if st.button("📝 Export Summaries"):
            json_path = export_summaries(st.session_state.summaries, 
                                       f"summaries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            if json_path:
                st.success(f"Summaries exported to: {json_path}")
        
        if st.button("🔑 Export Keywords"):
            json_path = export_summaries(st.session_state.keywords, 
                                       f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            if json_path:
                st.success(f"Keywords exported to: {json_path}")
    
    # Download processed data
    st.subheader("⬇️ Download Data")
    
    # Create downloadable CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    
    st.download_button(
        label="📄 Download CSV",
        data=csv_data,
        file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Create downloadable JSON
    json_data = {
        'documents': st.session_state.dataset,
        'summaries': st.session_state.summaries,
        'keywords': st.session_state.keywords,
        'export_date': datetime.now().isoformat()
    }
    
    st.download_button(
        label="📄 Download JSON",
        data=json.dumps(json_data, indent=2),
        file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
