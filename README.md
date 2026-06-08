# Research Paper Recommender & Subject Area Classifier 🧠📚

A dual-engine machine learning web application built with Streamlit that recommends academic papers based on semantic similarity and predicts the academic subject area of a given abstract.

## 🚀 Features
* **Semantic Recommendation Engine:** Uses a PyTorch-based Sentence Transformer (`all-MiniLM-L6-v2`) to generate document embeddings and perform highly accurate cosine-similarity searches to recommend relevant research papers.
* **Subject Area Classification:** Utilizes a custom-trained Multi-Layer Perceptron (MLP) with TF-IDF Text Vectorization in Keras/TensorFlow to classify abstracts into one of 165 distinct academic categories.
* **Apple Silicon Optimized:** Engineered a custom bypass for Google Colab Keras 3 serialization bugs (e.g., `quantization_config`) to allow seamless, native deployment on M-series Mac architecture without the need for cloud retraining.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Deep Learning/NLP:** PyTorch, TensorFlow, Keras, HuggingFace (`sentence-transformers`)
* **Data Processing:** NumPy, Pickle

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/Crazycraftie/Research-Paper-Recommendation.git]
cd [Your Repository Name]
2. Download the Pre-Trained Models
Because of GitHub's file size limits, the heavy AI models are hosted externally.

Download the model folder from here:

Place the model folder directly in the root directory of this project.

3. Install Dependencies
It is highly recommended to use a virtual environment.

Bash
pip install -r requirements.txt
4. Run the Application

Bash
streamlit run app.py
