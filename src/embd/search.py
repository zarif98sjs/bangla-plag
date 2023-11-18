import os
import json
import faiss
import numpy as np
import streamlit as st
import huggingface_hub as hub
from laserembeddings import Laser
from datasets import load_dataset
from annotated_text import annotated_text

hub.login(os.getenv("HUB_LOGIN"))
laser = Laser()

@st.cache_data
def load_dataset_from_huggingface():
    raw_dataset = load_dataset('zarif98sjs/bangla-plagiarism-dataset', use_auth_token=True)
    return raw_dataset

@st.cache_data
def load_contexts():
    contexts = raw_dataset["train"]['context']
    print("Number of contexts: ", len(contexts))
    contexts = list(set(contexts))
    print("Number of unique contexts: ", len(contexts))
    return sorted(contexts)

@st.cache_data
def load_embeddings():
    embds = np.load('embds.npy')
    print("Embeddings shape: ", embds.shape)
    return embds

@st.cache_data
def get_DOC_MAP():
    with open('DOC_MAP_START_IDX.json', 'r') as f:
        DOC_MAP = json.load(f)
    return DOC_MAP

@st.cache_data
def load_sentences():
    with open('all_sentences.txt', 'r',encoding='utf-8') as f:
        sentences = f.readlines()
    return sentences

def lower_bound_dict(d, target):
    # Get a sorted list of the values in the dictionary
    values = sorted(d.values())

    # Perform a binary search on the values to find the lower bound
    left = 1
    right = len(values) + 1
    while left != right:
        mid = (left + right) // 2
        if values[mid] <= target:
            left = mid + 1
        else:
            right = mid

    return left

def get_embd(sentence):
    return laser.embed_sentences(sentence,lang='bn')

@st.cache_data
def get_global_idx(embds):
    dimension = 1024
    index = faiss.IndexFlatL2(dimension)   # build the index
    print(index.is_trained)
    index.add(embds)                  # add vectors to the index
    print(index.ntotal)
    return index

raw_dataset = load_dataset_from_huggingface()
contexts = load_contexts()
all_sentences = load_sentences()
embds = load_embeddings()
DOC_MAP_START_IDX = get_DOC_MAP()
index = get_global_idx(embds)

def get_sentences_documents_pair(query_sentence,index):
    q_embd = get_embd(query_sentence)
    k = 3  # we want to see top 3 neighbors
    _, sentence_index = index.search(q_embd, k)
    sentence_index = sentence_index[0]
    document_index = []
    for s_idx in sentence_index:
        print(s_idx)
        document_index.append(lower_bound_dict(DOC_MAP_START_IDX,s_idx) - 1)
    return sentence_index, document_index


query = st.text_area("Enter your text here")
print(query)
sentence_idx, document_idx = get_sentences_documents_pair(query,index)
print("Sentence IDX:",sentence_idx)
print("Document IDX:",document_idx)

def get_sentences_for_markup(sent_idx_ind,doc_idx_ind,all_sentences):
    START_SENT_ID = DOC_MAP_START_IDX[str(doc_idx_ind + 1)]
    END_SENT_ID = DOC_MAP_START_IDX[str(doc_idx_ind + 2)]

    s_ara = []
    for s_idx in range(START_SENT_ID,END_SENT_ID):
        if sent_idx_ind == s_idx:
            s_ara.append((all_sentences[s_idx],"PLAG"))
        else:
            s_ara.append(all_sentences[s_idx])
    return s_ara

p1 = get_sentences_for_markup(sentence_idx[0],document_idx[0],all_sentences)
st.markdown("# First Match")
annotated_text(*p1)

p2 = get_sentences_for_markup(sentence_idx[1],document_idx[1],all_sentences)
st.markdown("# Second Match")
annotated_text(*p2)

p3 = get_sentences_for_markup(sentence_idx[2],document_idx[2],all_sentences)
st.markdown("# Third Match")
annotated_text(*p3)