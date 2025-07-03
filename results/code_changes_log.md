# Code Changes Log - Linguistic Features Implementation

## Overview
Technical implementation of linguistic feature support in the SVM classification pipeline. This extension enables comparison of simple interpretable features versus complex TF-IDF representations for medical text classification, with specific focus on overfitting assessment and clinical interpretability.

---

## File Modifications

### **1. `mk_doc_vectors.py` - MAJOR CHANGES**

#### **Added Function: `get_avg_sentence_length()`**
```python
def get_avg_sentence_length(text):
    """Calculate average sentence length in a text"""
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) == 0:
        return 0
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(word_counts)
```
**Purpose**: Computes average sentence length as linguistic feature for Alzheimer detection

#### **Added Variable: `current_text` accumulator**
```python
# Line ~59: Added after vecs = {}
current_text = ""  # To accumulate text of current document
```
**Purpose**: Accumulates full document text to compute linguistic features at document end

#### **Modified: Document start handling (`<doc` tag)**
```python
# Around line 65-75
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
```
**Changes**:
- Added `current_text = ""` reset for each document
- Added conditional vector initialization based on `feature_type`
- `linguistic`: 1D vector (1 feature only)
- `combined`: (vocab_size + 1)D vector (TF-IDF + linguistic)

#### **Modified: Document end handling (`</doc` tag)**
```python
# Around line 80-90
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
```
**Changes**:
- Added linguistic feature computation at document end
- `linguistic`: Sets single feature value
- `combined`: Normalizes TF-IDF part, then adds linguistic feature
- Preserves original normalization for other types

#### **Modified: Content processing (main text lines)**
```python
# Around line 95-115
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
```
**Changes**:
- Added `current_text += " " + l` to accumulate full document text
- Added `elif feature_type == "combined"` branch for TF-IDF processing
- `linguistic` type does no processing during text lines (computed at end)

#### **Modified: Output file naming**
```python
# Around line 125
vec_file = open(join(cat,f"vecs_{feature_type}.csv"),'w')
```
**Changes**:
- Dynamic filename based on `feature_type`
- Creates: `vecs_linguistic.csv`, `vecs_combined.csv`, etc.

---

### **2. `classification.py` - MINOR CHANGES**

#### **Updated: Help documentation**
```python
# Around line 4-12
"""
Usage:
  classification.py --C=<n> --kernel=[linear|poly|rbf] --features=[words|ngrams|linguistic|combined] [--degree=<n>]
  classification.py (-h | --help)
  classification.py --version

Options:
  --features=[words|ngrams|linguistic|combined]   Type of features to use.
"""
```
**Changes**:
- Updated docstring to include `linguistic` and `combined` options
- No code logic changes needed (reads from appropriate CSV files)

---

## Feature Types Behavior

### **`linguistic` (New)**
- **Vector size**: 1D (single feature)
- **Features**: `avg_sentence_length` only
- **Processing**: No vocabulary filtering - processes complete raw dataset
- **Overfitting risk**: Minimal due to single feature constraint
- **Output**: `vecs_linguistic.csv`

### **`combined` (New)**
- **Vector size**: (vocab_size + 1)D
- **Features**: Full TF-IDF + `avg_sentence_length` as last dimension  
- **Processing**: Standard TF-IDF pipeline + linguistic feature computation
- **Overfitting risk**: High due to high-dimensional space
- **Normalization**: Only TF-IDF features normalized, linguistic feature kept absolute
- **Output**: `vecs_combined.csv`

### **`ngrams` (Unchanged)**
- **Vector size**: vocab_size D
- **Features**: Character n-grams TF-IDF
- **Processing**: Original behavior preserved with vocabulary filtering
- **Overfitting risk**: High due to feature space size relative to data
- **Output**: `vecs_ngrams.csv`

### **`words` (Unchanged)**
- **Vector size**: vocab_size D  
- **Features**: Word-based TF-IDF
- **Processing**: Original behavior preserved
- **Output**: `vecs_words.csv`

---
