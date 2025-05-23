import streamlit as st
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import zipfile
import tempfile
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import io
import time
import re
import json
import csv
import requests
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
import difflib

# Try to import optional dependencies with fallbacks
try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except ImportError:
    KNEED_AVAILABLE = False
    st.warning("KneeLocator not available. Auto-clustering will use fallback method.")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    
    # Download NLTK resources (wrapped in try-except to handle offline scenarios)
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception:
        pass
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Some text processing features will be limited.")

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

# Import TensorFlow components
try:
        Tokenizer = tf.keras.preprocessing.text.Tokenizer
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
except AttributeError:
        st.error("TensorFlow Keras preprocessing not available")

# ----------- Helper Functions -----------

def get_tesseract_path():
    """Handle Tesseract path based on platform"""
    if os.name == 'nt':  # Windows
        return r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:  # Unix/Linux/MacOS
        return pytesseract.pytesseract.tesseract_cmd

# Try to set Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()
except Exception:
    st.warning("Tesseract OCR path not configured. Please install Tesseract OCR.")

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
    model = make_pipeline(CountVectorizer(max_features=5000), MultinomialNB())
    model.fit(X, y)
    return model

def create_rnn_model(input_length, num_classes, embedding_dim=128):
    """Create an RNN model with improved architecture"""
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

# ----------- Text Extraction -----------

@safe_process
def extract_text_from_image(image, lang='eng'):
    """Extract text from an image with language support"""
    try:
        return pytesseract.image_to_string(image, lang=lang)
    except Exception as e:
        st.warning(f"OCR extraction failed: {e}")
        return ""

@safe_process
def extract_text_from_pdf(pdf_file, lang='eng'):
    """Extract text from PDF with language support"""
    try:
        images = convert_from_path(pdf_file)
        return "\n".join(extract_text_from_image(image, lang=lang) for image in images)
    except Exception as e:
        st.warning(f"PDF processing failed: {e}")
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

def simple_sent_tokenize(text):
    """Simple sentence tokenizer as fallback"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_summary(text, max_sentences=5):
    """Generate a summary using extractive summarization"""
    if not text or len(text) < 100:
        return text
    
    try:
        # Split text into sentences
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            sentences = simple_sent_tokenize(text)
        
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
    individual_texts = {}  # Store text for each file separately
    summaries = {}  # Store summaries for each file
    keywords = {}  # Store keywords for each file
    
    for filepath in file_list:
        filename = os.path.basename(filepath)
        # Use the filename without extension as the label
        label = os.path.splitext(filename)[0]
        
        try:
            text = ""
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(filepath, lang=language)
            else:
                text = extract_text_from_image(Image.open(filepath), lang=language)
            
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
                
                # Store individual text for each file
                individual_texts[filename] = text.strip()
                
                # Store summary and keywords
                summaries[filename] = summary
                keywords[filename] = doc_keywords
                
                # Add to combined text
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
            # Concatenate and remove duplicates based on filename
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
    # Check if we have enough data
    if len(df) < 2 or len(df["label"].unique()) < 2:
        st.warning("⚠️ Not enough data or unique labels to train model")
        return None, None, None, None, None
    
    unique_labels = df["label"].unique()
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = df["label"].map(label_map)
    texts = df["text"].fillna("").tolist()
    
    with st.spinner("Training Naive Bayes model..."):
        try:
            nb_model = train_naive_bayes(texts, y)
        except Exception as e:
            st.error(f"Error training Naive Bayes model: {e}")
            return None, None, None, None, None
    
    # Try to train RNN model
    rnn_model = None
    tokenizer = None
    max_length = 100
    
    try:
        with st.spinner("Preparing RNN data..."):
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(texts)
            sequences = tokenizer.texts_to_sequences(texts)
            max_length = min(max([len(seq) for seq in sequences]) if sequences else 100, 500)
            padded_seq = pad_sequences(sequences, maxlen=max_length)
        
        with st.spinner("Training RNN model..."):
            rnn_model = create_rnn_model(max_length, len(unique_labels))
            # Add early stopping
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
        st.warning(f"RNN model training failed: {e}. Continuing with Naive Bayes only.")
        rnn_model = None
        tokenizer = None
    
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
    
    # Get max sequence length from metadata if available
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
            
            # Add confidence scores for RNN predictions
            confidence_scores = np.max(rnn_preds, axis=1)
            
            result_df["RNN_Label"] = rnn_labels
            result_df["RNN_Confidence"] = [f"{score:.2%}" for score in confidence_scores]
        except Exception as e:
            st.error(f"Error in RNN prediction: {e}")
            result_df["RNN_Label"] = ["Error" for _ in range(len(texts))]
            result_df["RNN_Confidence"] = ["N/A" for _ in range(len(texts))]
    else:
        # If RNN model is not available, use only Naive Bayes
        result_df["RNN_Label"] = ["N/A" for _ in range(len(texts))]
        result_df["RNN_Confidence"] = ["N/A" for _ in range(len(texts))]
    
    return result_df

def load_saved_models():
    """Load all saved models if they exist"""
    nb_model = load_model(NB_MODEL_PATH)
    metadata = load_model(METADATA_PATH)
    rnn_model = None
    
    if os.path.exists(RNN_MODEL_PATH):
        try:
            rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)
        except Exception as e:
            st.warning(f"Could not load RNN model: {e}")
    
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
            
        # Simple search first
        if query in text.lower():
            # Calculate relevance score based on frequency and position
            frequency = text.lower().count(query)
            position = text.lower().find(query) / len(text) if len(text) > 0 else 1
            score = (frequency * 0.7) + ((1 - position) * 0.3)
            
            # Get context around the query
            idx = text.lower().find(query)
            start = max(0, idx - 100)
            end = min(len(text), idx + len(query) + 100)
            context = text[start:end]
            
            # Highlight the query in context
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
                
            results.append({
                "filename": filename,
                "score": min(score, 1.0),  # Cap at 1.0
                "context": context,
                "match_type": "direct"
            })
        else:
            # For documents without direct matches, use TF-IDF similarity
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
                # If vectorization fails (e.g., empty text), skip
                continue
    
    # Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ----------- K-Means Clustering Functions -----------

def get_optimal_clusters(vectorized_data, max_k=10):
    """Find optimal number of clusters using Elbow method"""
    # Make sure max_k doesn't exceed number of samples
    n_samples = vectorized_data.shape[0]
    max_k = min(max_k, n_samples - 1)  # Ensure max_k is at most n_samples-1
    
    wcss = []  # Within-cluster sum of squares
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(vectorized_data)
        wcss.append(kmeans.inertia_)
    
    # Find elbow point
    if KNEED_AVAILABLE:
        try:
            kl = KneeLocator(range(1, max_k + 1), wcss, curve="convex", direction="decreasing")
            optimal_k = kl.elbow if kl.elbow else min(3, max_k)  # Default to 3 or max_k if smaller
        except Exception as e:
            st.warning(f"Error finding optimal clusters: {e}")
            optimal_k = min(3, max_k)  # Fallback to 3 clusters if KneeLocator fails
    else:
        # Simple elbow detection fallback
        if len(wcss) >= 3:
            # Find the point with maximum decrease in slope
            diffs = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
            optimal_k = diffs.index(max(diffs)) + 2  # +2 because we start from k=1 and need the k value
            optimal_k = min(optimal_k, max_k)
        else:
            optimal_k = min(3, max_k)
    
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
        perplexity = min(30, vectorized_data.shape[0] - 1)
        if perplexity < 1:
            perplexity = 1
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity)
        return reducer.fit_transform(vectorized_data.toarray())
    else:
        # Fallback to PCA for unsupported methods
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(vectorized_data.toarray())

def plot_elbow_curve(wcss):
    """Plot elbow curve to help find optimal K"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(range(1, len(wcss) + 1), wcss, 'bo-')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS (Within-Cluster Sum of Square)')
    plt.grid(True)
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

def get_top_cluster_terms(vectorizer, kmeans, n_terms=10):
    """Get top terms characterizing each cluster"""
    # Get cluster centers
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_terms = {}
    for i in range(kmeans.n_clusters):
        cluster_terms[i] = [terms[ind] for ind in order_centroids[i, :n_terms]]
    
    return cluster_terms

def export_results(df, format_type='csv'):
    """Export results to different formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == 'csv':
        filename = f"document_analysis_{timestamp}.csv"
        filepath = os.path.join(EXPORT_DIR, filename)
        df.to_csv(filepath, index=False)
        return filepath
    elif format_type == 'json':
        filename = f"document_analysis_{timestamp}.json"
        filepath = os.path.join(EXPORT_DIR, filename)
        df.to_json(filepath, orient='records', indent=2)
        return filepath
    elif format_type == 'excel':
        filename = f"document_analysis_{timestamp}.xlsx"
        filepath = os.path.join(EXPORT_DIR, filename)
        df.to_excel(filepath, index=False)
        return filepath
    else:
        return None

def create_analysis_report(df, summaries, keywords, comparison_results=None):
    """Create a comprehensive analysis report"""
    report = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_documents": len(df),
        "document_statistics": {
            "avg_char_count": df['char_count'].mean() if 'char_count' in df.columns else 0,
            "avg_word_count": df['word_count'].mean() if 'word_count' in df.columns else 0,
            "languages_detected": df['language'].unique().tolist() if 'language' in df.columns else [],
            "labels_found": df['label'].unique().tolist() if 'label' in df.columns else []
        },
        "summaries": summaries,
        "keywords": keywords,
        "comparison_results": comparison_results or {}
    }
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"analysis_report_{timestamp}.json"
    report_filepath = os.path.join(SUMMARY_DIR, report_filename)
    
    with open(report_filepath, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report, report_filepath

# ----------- Streamlit UI -----------

def main():
    st.set_page_config(
        page_title="Advanced Document Analysis & OCR",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📄 Advanced Document Analysis & OCR System")
    st.markdown("---")
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("📚 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["📄 Document Processing", "🤖 Model Training", "🔍 Document Search", 
             "📊 Clustering Analysis", "📈 Analytics Dashboard", "⚙️ Settings"]
        )
        
        st.header("🌐 Language Settings")
        language = st.selectbox(
            "OCR Language:",
            ["eng", "spa", "fra", "deu", "ita", "por", "rus", "chi_sim", "jpn", "kor"],
            help="Select the primary language for OCR processing"
        )
        
        st.header("📊 Model Status")
        # Check if models exist
        nb_exists = os.path.exists(NB_MODEL_PATH)
        rnn_exists = os.path.exists(RNN_MODEL_PATH)
        
        st.write(f"Naive Bayes: {'✅' if nb_exists else '❌'}")
        st.write(f"RNN Model: {'✅' if rnn_exists else '❌'}")
        
        # Load existing training data stats
        training_df = load_training_data()
        if not training_df.empty:
            st.write(f"Training samples: {len(training_df)}")
            st.write(f"Unique labels: {len(training_df['label'].unique())}")
    
    # Main content based on selected page
    if page == "📄 Document Processing":
        document_processing_page(language)
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "🔍 Document Search":
        document_search_page()
    elif page == "📊 Clustering Analysis":
        clustering_analysis_page()
    elif page == "📈 Analytics Dashboard":
        analytics_dashboard_page()
    elif page == "⚙️ Settings":
        settings_page()

def document_processing_page(language):
    """Main document processing page"""
    st.header("📄 Document Processing")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["📁 Upload Files", "🗜️ Upload ZIP", "📂 Local Folder"],
        horizontal=True
    )
    
    file_list = []
    
    if input_method == "📁 Upload Files":
        uploaded_files = st.file_uploader(
            "Upload images or PDFs",
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'pdf', 'tiff', 'bmp']
        )
        if uploaded_files:
            file_list = get_files_from_upload(uploaded_files)
    
    elif input_method == "🗜️ Upload ZIP":
        uploaded_zip = st.file_uploader(
            "Upload ZIP file containing images/PDFs",
            type=['zip']
        )
        if uploaded_zip:
            file_list = get_files_from_zip(uploaded_zip)
    
    elif input_method == "📂 Local Folder":
        folder_path = st.text_input("Enter folder path:")
        if folder_path and st.button("Load Files"):
            file_list = get_files_from_folder(folder_path)
    
    if file_list:
        st.success(f"Found {len(file_list)} files to process")
        
        # Processing options
        col1, col2 = st.columns(2)
        with col1:
            process_files_btn = st.button("🚀 Process Files", type="primary")
        with col2:
            save_to_training = st.checkbox("Save to training data", value=True)
        
        if process_files_btn:
            with st.spinner("Processing files..."):
                dataset, all_text, classification_counts, individual_texts, summaries, keywords = process_files(file_list, language)
            
            if dataset:
                df = pd.DataFrame(dataset)
                st.success(f"✅ Successfully processed {len(dataset)} documents")
                
                # Display results
                st.subheader("📊 Processing Results")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Documents", len(dataset))
                with col2:
                    st.metric("Unique Labels", len(classification_counts))
                with col3:
                    st.metric("Avg Words/Doc", int(df['word_count'].mean()))
                with col4:
                    st.metric("Avg Chars/Doc", int(df['char_count'].mean()))
                
                # Document table
                st.subheader("📋 Document Details")
                display_df = df[['filename', 'label', 'word_count', 'char_count', 'processed_date']]
                st.dataframe(display_df, use_container_width=True)
                
                # Individual document viewer
                st.subheader("📖 Document Viewer")
                selected_file = st.selectbox("Select document to view:", list(individual_texts.keys()))
                
                if selected_file:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Full Text:**")
                        st.text_area("", individual_texts[selected_file], height=300)
                    
                    with col2:
                        st.write("**Summary:**")
                        st.text_area("", summaries.get(selected_file, "No summary available"), height=150)
                        
                        st.write("**Keywords:**")
                        if selected_file in keywords and keywords[selected_file]:
                            for keyword, score in keywords[selected_file][:10]:
                                st.write(f"• {keyword} ({score:.3f})")
                        else:
                            st.write("No keywords extracted")
                
                # Export options
                st.subheader("💾 Export Options")
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    if st.button("Export CSV"):
                        filepath = export_results(df, 'csv')
                        if filepath:
                            st.success(f"Exported to {filepath}")
                
                with export_col2:
                    if st.button("Export JSON"):
                        filepath = export_results(df, 'json')
                        if filepath:
                            st.success(f"Exported to {filepath}")
                
                with export_col3:
                    if st.button("Create Report"):
                        report, report_path = create_analysis_report(df, summaries, keywords)
                        st.success(f"Report created: {report_path}")
                
                # Save to training data
                if save_to_training:
                    if save_training_data(df):
                        st.success("📚 Data saved to training dataset")
                    else:
                        st.error("❌ Failed to save training data")
                
                # Store in session state for other pages
                st.session_state['current_df'] = df
                st.session_state['individual_texts'] = individual_texts
                st.session_state['summaries'] = summaries
                st.session_state['keywords'] = keywords

def model_training_page():
    """Model training and prediction page"""
    st.header("🤖 Model Training & Prediction")
    
    # Load existing training data
    training_df = load_training_data()
    
    if training_df.empty:
        st.warning("⚠️ No training data available. Please process some documents first.")
        return
    
    # Display training data statistics
    st.subheader("📊 Training Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(training_df))
    with col2:
        st.metric("Unique Labels", len(training_df['label'].unique()))
    with col3:
        st.metric("Avg Words/Doc", int(training_df['word_count'].mean()))
    
    # Show label distribution
    st.subheader("📈 Label Distribution")
    label_counts = training_df['label'].value_counts()
    fig = px.bar(x=label_counts.index, y=label_counts.values, title="Document Count by Label")
    st.plotly_chart(fig, use_container_width=True)
    
    # Training controls
    st.subheader("🏋️ Model Training")
    
    col1, col2 = st.columns(2)
    with col1:
        train_btn = st.button("🚀 Train Models", type="primary")
    with col2:
        auto_save = st.checkbox("Auto-save models", value=True)
    
    if train_btn:
        with st.spinner("Training models... This may take a few minutes."):
            nb_model, rnn_model, tokenizer, label_map, unique_labels = train_models(training_df, auto_save)
        
        if nb_model:
            st.success("✅ Models trained successfully!")
            
            # Store models in session state
            st.session_state['nb_model'] = nb_model
            st.session_state['rnn_model'] = rnn_model
            st.session_state['tokenizer'] = tokenizer
            st.session_state['label_map'] = label_map
            st.session_state['unique_labels'] = unique_labels
        else:
            st.error("❌ Model training failed")
    
    # Prediction section
    st.subheader("🔮 Make Predictions")
    
    # Check if we have models
    if 'nb_model' in st.session_state or os.path.exists(NB_MODEL_PATH):
        # Load models if not in session state
        if 'nb_model' not in st.session_state:
            nb_model, rnn_model, tokenizer, label_map, unique_labels, metadata = load_saved_models()
            if nb_model:
                st.session_state['nb_model'] = nb_model
                st.session_state['rnn_model'] = rnn_model
                st.session_state['tokenizer'] = tokenizer
                st.session_state['label_map'] = label_map
                st.session_state['unique_labels'] = unique_labels
                st.session_state['metadata'] = metadata
        
        # Prediction input
        prediction_text = st.text_area("Enter text for prediction:", height=150)
        
        if prediction_text and st.button("🎯 Predict"):
            # Create temporary dataframe for prediction
            pred_df = pd.DataFrame({
                'filename': ['user_input'],
                'text': [prediction_text],
                'label': ['unknown']
            })
            
            # Make prediction
            result_df = predict_with_models(
                pred_df,
                st.session_state['nb_model'],
                st.session_state.get('rnn_model'),
                st.session_state.get('tokenizer'),
                st.session_state.get('label_map'),
                st.session_state.get('unique_labels'),
                st.session_state.get('metadata')
            )
            
            # Display results
            st.subheader("🎯 Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Naive Bayes Prediction:**")
                st.write(f"Label: {result_df['NaiveBayes_Label'].iloc[0]}")
            
            with col2:
                st.write("**RNN Prediction:**")
                st.write(f"Label: {result_df['RNN_Label'].iloc[0]}")
                st.write(f"Confidence: {result_df['RNN_Confidence'].iloc[0]}")
    else:
        st.info("ℹ️ Train models first to enable predictions")

def document_search_page():
    """Document search page"""
    st.header("🔍 Document Search")
    
    if 'individual_texts' not in st.session_state:
        st.warning("⚠️ No documents loaded. Please process documents first.")
        return
    
    # Search interface
    search_query = st.text_input("🔍 Enter search query:", placeholder="Search for keywords, phrases, or concepts...")
    
    col1, col2 = st.columns(2)
    with col1:
        search_btn = st.button("🔍 Search", type="primary")
    with col2:
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.1)
    
    if search_query and search_btn:
        # Perform search
        results = search_documents(st.session_state['individual_texts'], search_query, threshold)
        
        if results:
            st.success(f"Found {len(results)} matching documents")
            
            # Display results
            for i, result in enumerate(results):
                with st.expander(f"📄 {result['filename']} (Score: {result['score']:.3f})"):
                    st.write(f"**Match Type:** {result['match_type'].title()}")
                    st.write(f"**Relevance Score:** {result['score']:.3f}")
                    st.write("**Context:**")
                    st.text_area("", result['context'], height=100, key=f"context_{i}")
        else:
            st.info("No matching documents found. Try adjusting the search query or threshold.")

def clustering_analysis_page():
    """Document clustering analysis page"""
    st.header("📊 Document Clustering Analysis")
    
    # Check if we have current data
    if 'current_df' not in st.session_state:
        st.warning("⚠️ No documents loaded. Please process documents first.")
        return
    
    df = st.session_state['current_df']
    
    if len(df) < 2:
        st.warning("⚠️ Need at least 2 documents for clustering analysis.")
        return
    
    # Clustering controls
    st.subheader("⚙️ Clustering Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        auto_clusters = st.checkbox("Auto-detect optimal clusters", value=True)
        if not auto_clusters:
            n_clusters = st.slider("Number of clusters", 2, min(10, len(df)), 3)
    with col2:
        dim_reduction = st.selectbox("Dimension Reduction", ["pca", "svd", "tsne"])
    
    if st.button("🚀 Perform Clustering Analysis", type="primary"):
        with st.spinner("Performing clustering analysis..."):
            # Create vectorizer and vectorize documents
            vectorizer, vectorized_data = create_vectorizer(df)
            
            # Determine optimal number of clusters
            if auto_clusters:
                optimal_k, wcss = get_optimal_clusters(vectorized_data, max_k=min(10, len(df)-1))
                st.info(f"Optimal number of clusters detected: {optimal_k}")
            else:
                optimal_k = n_clusters
                _, wcss = get_optimal_clusters(vectorized_data, max_k=min(10, len(df)-1))
            
            # Train K-means
            kmeans = train_kmeans(vectorized_data, optimal_k)
            
            # Add cluster labels to dataframe
            df_clustered = df.copy()
            df_clustered['Cluster'] = kmeans.labels_
            
            # Results display
            st.subheader("📊 Clustering Results")
            
            # Cluster statistics
            cluster_stats = df_clustered['Cluster'].value_counts().sort_index()
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Cluster Distribution:**")
                for cluster_id, count in cluster_stats.items():
                    st.write(f"Cluster {cluster_id}: {count} documents")
            
            with col2:
                # Elbow curve
                st.write("**Elbow Curve:**")
                elbow_fig = plot_elbow_curve(wcss)
                st.pyplot(elbow_fig)
            
            # Visualization
            st.subheader("📈 Cluster Visualization")
            
            # 2D visualization
            reduced_data_2d = reduce_dimensions(vectorized_data, dim_reduction, 2)
            cluster_fig_2d = plot_clusters_2d(df_clustered, reduced_data_2d, kmeans)
            st.plotly_chart(cluster_fig_2d, use_container_width=True)
            
            # 3D visualization
            if len(df) > 3:
                reduced_data_3d = reduce_dimensions(vectorized_data, dim_reduction, 3)
                cluster_fig_3d = plot_clusters_3d(df_clustered, reduced_data_3d, kmeans)
                st.plotly_chart(cluster_fig_3d, use_container_width=True)
            
            # Top terms per cluster
            st.subheader("🏷️ Cluster Characteristics")
            cluster_terms = get_top_cluster_terms(vectorizer, kmeans, n_terms=10)
            
            for cluster_id, terms in cluster_terms.items():
                st.write(f"**Cluster {cluster_id} Top Terms:** {', '.join(terms)}")
            
            # Detailed cluster view
            st.subheader("📋 Documents by Cluster")
            selected_cluster = st.selectbox("Select cluster to view:", sorted(df_clustered['Cluster'].unique()))
            
            cluster_docs = df_clustered[df_clustered['Cluster'] == selected_cluster]
            st.dataframe(cluster_docs[['filename', 'label', 'word_count']], use_container_width=True)

def analytics_dashboard_page():
    """Analytics dashboard page"""
    st.header("📈 Analytics Dashboard")
    
    # Load all available data
    training_df = load_training_data()
    
    if training_df.empty:
        st.warning("⚠️ No data available for analytics.")
        return
    
    # Overall statistics
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(training_df))
    with col2:
        st.metric("Unique Labels", len(training_df['label'].unique()))
    with col3:
        st.metric("Avg Words/Doc", int(training_df['word_count'].mean()))
    with col4:
        st.metric("Processing Languages", len(training_df['language'].unique()))
    
    # Time series analysis
    if 'processed_date' in training_df.columns:
        st.subheader("📅 Processing Timeline")
        training_df['processed_date'] = pd.to_datetime(training_df['processed_date'])
        daily_counts = training_df.groupby(training_df['processed_date'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        timeline_fig = px.line(daily_counts, x='date', y='count', title="Documents Processed Over Time")
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Word count distribution
    st.subheader("📊 Word Count Distribution")
    word_count_fig = px.histogram(training_df, x='word_count', title="Distribution of Document Word Counts")
    st.plotly_chart(word_count_fig, use_container_width=True)
    
    # Label analysis
    st.subheader("🏷️ Label Analysis")
    label_stats = training_df['label'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        pie_fig = px.pie(values=label_stats.values, names=label_stats.index, title="Document Distribution by Label")
        st.plotly_chart(pie_fig, use_container_width=True)
    
    with col2:
        bar_fig = px.bar(x=label_stats.index, y=label_stats.values, title="Document Count by Label")
        st.plotly_chart(bar_fig, use_container_width=True)

def settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")
    
    # Model management
    st.subheader("🤖 Model Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All Models"):
            model_files = [NB_MODEL_PATH, RNN_MODEL_PATH, METADATA_PATH, KMEANS_MODEL_PATH, VECTORIZER_PATH]
            for file_path in model_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
            st.success("✅ All models cleared")
    
    with col2:
        if st.button("🗑️ Clear Training Data"):
            if os.path.exists(TRAINING_DATA_PATH):
                os.remove(TRAINING_DATA_PATH)
            st.success("✅ Training data cleared")
    
    # Directory information
    st.subheader("📁 Directory Information")
    directories = {
        "Models": MODEL_DIR,
        "Training Data": DATA_DIR,
        "Exports": EXPORT_DIR,
        "Summaries": SUMMARY_DIR
    }
    
    for name, path in directories.items():
        file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]) if os.path.exists(path) else 0
        st.write(f"**{name}:** {path} ({file_count} files)")
    
    # System info
    st.subheader("💻 System Information")
    st.write(f"**TensorFlow Version:** {tf.__version__}")
    st.write(f"**NLTK Available:** {'✅' if NLTK_AVAILABLE else '❌'}")
    st.write(f"**KneeLocator Available:** {'✅' if KNEED_AVAILABLE else '❌'}")
    
    # Advanced settings
    st.subheader("🔧 Advanced Settings")
    
    # Clear session state
    if st.button("🔄 Clear Session State"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Session state cleared")
    
    # Export all data
    if st.button("📦 Export All Data"):
        training_df = load_training_data()
        if not training_df.empty:
            export_path = export_results(training_df, 'csv')
            st.success(f"✅ All data exported to {export_path}")
        else:
            st.warning("⚠️ No data to export")

if __name__ == "__main__":
    main()
