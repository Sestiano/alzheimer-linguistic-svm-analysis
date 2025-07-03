import re
import sys
import numpy as np
from math import isnan
from os import listdir
from os.path import isfile, isdir, join

feature_type = sys.argv[1]    

def get_avg_sentence_length(text):
    """Calculate average sentence length in a text"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) == 0:
        return 0
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)

def read_vocab():
    with open('./data/vocab_file.txt','r') as f:
        vocab = f.read().splitlines()
    return vocab

def get_words(l):
    l=l.lower()
    words = {}
    for word in l.split():
        if word in words:
            words[word]+=1
        else:
            words[word]=1
    return words

def get_ngrams(l,n):
    l = l.lower()
    ngrams = {}
    for i in range(0,len(l)-n+1):
        ngram = l[i:i+n]
        if ngram in ngrams:
            ngrams[ngram]+=1
        else:
            ngrams[ngram]=1
    return ngrams

def normalise(v):
    return v / sum(v)

def clean_docs(d,docs):
    m = []
    retained_docs = []
    for url in docs:
        if not isnan(sum(d[url])) and sum(d[url]) != 0:
            m.append(d[url])
            retained_docs.append(url)
    return np.array(m), retained_docs



d = './data'
catdirs = [join(d,o) for o in listdir(d) if isdir(join(d,o))]
vocab = read_vocab()

for cat in catdirs:
    print(cat)
    url = ""
    docs = []
    vecs = {}
    current_text = ""  # To accumulate text of current document
    doc_file = open(join(cat,"linear.txt"),'r')
    for l in doc_file:
        l=l.rstrip('\n')
        if l[:4] == "<doc":
            m = re.search("date=(.*)>",l)
            url = m.group(1).replace(',',' ')
            docs.append(url)
            current_text = ""  # Reset for new document
            if feature_type == "linguistic":
                vecs[url] = np.zeros(1)  # Only 1 feature
            elif feature_type == "combined":
                vecs[url] = np.zeros(len(vocab) + 1)  # TF-IDF + 1 linguistic
            else:
                vecs[url] = np.zeros(len(vocab))
        elif l[:5] == "</doc":
            # Calculate linguistic feature at document end
            if feature_type == "linguistic":
                vecs[url][0] = get_avg_sentence_length(current_text)
            elif feature_type == "combined":
                vecs[url] = normalise(vecs[url][:-1])  # Normalize only TF-IDF features
                vecs[url][-1] = get_avg_sentence_length(current_text)
            else:
                vecs[url] = normalise(vecs[url])
            print(url)
        else:
            current_text += " " + l  # Accumulate text
            
            if feature_type == "ngrams":
                for i in range(3,7): #hacky...
                    ngrams = get_ngrams(l,i)
                    for k,v in ngrams.items():
                        if k in vocab:
                            vecs[url][vocab.index(k)]+=v
            elif feature_type == "words":
                words = get_words(l)
                for k,v in words.items():
                    if k in vocab:
                        vecs[url][vocab.index(k)]+=v
            elif feature_type == "combined":
                # For combined, do both ngrams and linguistic feature
                for i in range(3,7):
                    ngrams = get_ngrams(l,i)
                    for k,v in ngrams.items():
                        if k in vocab:
                            vecs[url][vocab.index(k)]+=v
            # For "linguistic", do nothing here - calculate at the end
                 
        
    
    doc_file.close()
    m,retained_docs = clean_docs(vecs,docs)
    print("------------------")
    print("NUM ORIGINAL DOCS:", len(docs))
    print("NUM RETAINED DOCS:", len(retained_docs))
 
    vec_file = open(join(cat,f"vecs_{feature_type}.csv"),'w')
    for i in range(len(retained_docs)):
            vec_file.write(retained_docs[i]+','+','.join([str(v) for v in m[i]])+'\n') 
    vec_file.close()
