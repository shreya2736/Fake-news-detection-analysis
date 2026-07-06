# Fake News Detection using Knowledge Graphs and TF-IDF

## Overview

This project implements a **hybrid fake news detection system** that combines:

- Natural Language Processing (NLP)
- Named Entity Recognition (NER)
- Knowledge Graph construction
- TF-IDF text representation
- Logistic Regression classification

Instead of relying only on textual features, the model also captures relationships between named entities using a knowledge graph, improving the understanding of contextual information within news statements.


## Features

- Text preprocessing and cleaning
- Named Entity Recognition using spaCy
- Knowledge Graph generation using NetworkX
- Entity-based graph scoring
- TF-IDF feature extraction
- Logistic Regression classifier
- Graph visualization of important entities
- Binary fake news classification
  

## Dataset

The project uses the **LIAR Dataset**, which contains political statements labeled with six truthfulness categories.

Original labels:

- Pants-fire
- False
- Barely-true
- Half-true
- Mostly-true
- True

For this project, these labels are converted into binary classes:

| Original Label | Binary Label |
|---------------|--------------|
| Pants-fire | False |
| False | False |
| Barely-true | False |
| Half-true | True |
| Mostly-true | True |
| True | True |


## Technologies Used

- Python
- Pandas
- spaCy
- NetworkX
- Scikit-learn
- Matplotlib
- SciPy


## Project Workflow

### 1. Data Loading

The LIAR dataset is loaded into Pandas DataFrames for training, testing, and validation.


### 2. Text Preprocessing

Each statement undergoes preprocessing:

- Convert to lowercase
- Remove punctuation
- Remove numbers
- Remove extra whitespace


### 3. Named Entity Recognition

spaCy's `en_core_web_sm` model extracts entities such as:

- PERSON
- ORG
- GPE

Example:

Statement:

> Barack Obama visited Washington.

Extracted entities:

- Barack Obama
- Washington


### 4. Knowledge Graph Construction

Using NetworkX:

- Every extracted entity becomes a node.
- Entities appearing in the same statement are connected by edges.
- Entity credibility scores are calculated based on training labels.

True statements increase an entity's score, while false statements decrease it.


### 5. Graph-Based Feature

Each statement receives a graph score by averaging the credibility scores of its extracted entities.

This score provides contextual information beyond the raw text.


### 6. TF-IDF Feature Extraction

The cleaned statements are transformed into numerical vectors using TF-IDF.

Maximum vocabulary size:

- 500 features


### 7. Feature Combination

The TF-IDF vectors are combined with the graph score to create a richer feature representation.

Combined Features:

- TF-IDF vector
- Graph credibility score


### 8. Classification

A Logistic Regression classifier is trained using the combined features.

Performance is evaluated using classification accuracy.


### 9. Graph Visualization

The project visualizes the most connected entities in the knowledge graph using NetworkX and Matplotlib.

This helps understand relationships between frequently occurring entities.


## Project Structure

```
├── codefile.ipynb
├── train.tsv
├── test.tsv
├── valid.tsv
├── README.md
```


## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-knowledge-graph.git

cd fake-news-knowledge-graph
```

Install dependencies:

```bash
pip install pandas
pip install spacy
pip install networkx
pip install matplotlib
pip install scikit-learn
pip install scipy
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```


## Running the Project

Open the notebook:

```bash
jupyter notebook codefile.ipynb
```

Run all cells sequentially.



## Output

The notebook provides:

- Binary fake news predictions
- Model accuracy
- Knowledge graph visualization
- Top connected entities



## Future Improvements

- Use transformer-based embeddings (BERT/RoBERTa)
- Apply Graph Neural Networks (GNNs)
- Incorporate fact-checking APIs
- Multi-class classification instead of binary labels
- Deploy as a web application using Flask or Streamlit



## Learning Outcomes

This project demonstrates how to combine traditional NLP techniques with graph-based representations for fake news detection. By integrating entity relationships with textual features, the model captures contextual information that may improve classification performance.


