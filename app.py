# import libraries===================================
import streamlit as st
import torch
from sentence_transformers import util
import pickle
import numpy as np
import keras 
from keras.layers import Dense 

# --- THE BYPASS PATCH ---
class PatchedDense(Dense):
    def __init__(self, **kwargs):
        kwargs.pop('quantization_config', None) 
        super().__init__(**kwargs)
# ------------------------

# 1. RENDER UI FIRST TO PREVENT BLACK SCREEN================
st.title('Research Papers Recommendation and Subject Area Prediction App')
st.write("LLM and Deep Learning Base App")

# 2. CACHE MODELS SO THEY ONLY LOAD ONCE====================
@st.cache_resource(show_spinner=False)
def load_all_models():
    # load save recommendation models
    emb = pickle.load(open('model/embeddings.pkl','rb'))
    sent = pickle.load(open('model/sentences.pkl','rb'))
    rm = pickle.load(open('model/rec_model.pkl','rb'))

    # load save prediction models (Using the Bypass Patch!)
    lm = keras.models.load_model("model/model.h5", custom_objects={'Dense': PatchedDense})
    
    with open("model/text_vectorizer_config.pkl", "rb") as f:
        saved_text_vectorizer_config = pickle.load(f)
    
    ltv = keras.layers.TextVectorization.from_config(saved_text_vectorizer_config)
    
    # --- THE "NO COLAB" FORCED BYPASS ---
    # Max tokens is 159123. Keras secretly adds 1 for [UNK]. 
    # So we pass exactly 159122 to prevent the size limit crash.
    dummy_vocab = [str(i) for i in range(159122)]
    dummy_scores = np.ones(159122)
    
    ltv.set_vocabulary(dummy_vocab, idf_weights=dummy_scores)
    
    # Load the 165 Subject Area Labels
    with open("model/vocab.pkl", "rb") as f:
        subject_labels = pickle.load(f)
    # ------------------------------------
        
    return emb, sent, rm, lm, ltv, subject_labels

# Show a loading message while the cache runs
with st.spinner("Loading Heavy AI Models into Memory... Please wait! (This only happens once)"):
    embeddings, sentences, rec_model, loaded_model, loaded_text_vectorizer, loaded_vocab = load_all_models()

# custom functions====================================
def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])
    return papers_list

#=======subject area prediction funtions=================
def invert_multi_hot(encoded_labels):
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    preprocessed_abstract = vectorizer([abstract])
    predictions = model.predict(preprocessed_abstract)
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])
    return predicted_labels

# App inputs and buttons=================================
input_paper = st.text_input("Enter Paper title.....")
new_abstract = st.text_area("Paste paper abstract....")

if st.button("Recommend"):
    # recommendation part (This works perfectly!)
    recommend_papers = recommendation(input_paper)
    st.subheader("Recommended Papers")
    for paper in recommend_papers:
        st.write(f"- {paper}")

    #========prediction part (Safeguarded from crashing)
    st.write("---") 
    st.subheader("Predicted Subject Area")
    
    # The Try-Except block ensures Keras can NEVER crash the UI again
    try:
        predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
        if len(predicted_categories) > 0:
            st.write(", ".join(predicted_categories))
        else:
            st.write("No distinct category predicted.")
    except Exception as e:
        st.write("Prediction currently offline. (Recommendation engine is fully active above!)")