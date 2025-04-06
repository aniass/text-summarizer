# Text Summarization

## General info
Text summarization based on extractive and abstractive methods by using python. In this project I have presented three examples of the extractive technique such as calculating word frequency with spacy library, TFIDF vectorizer implementation and automatic text summarization with gensim library. To show abstractive techniques I have used Hugging Face Transformer library and BART model.

### Dataset
In this project I have used an article coming from [BBC news](https://www.bbc.com/news). This downloaded data one can see in data folder in text file.

## Motivation
One of the application of text analysis and NLP is Text Summarization. It is a technique of shortening long pieces of text into a short message. The intention is to create a cohesive and fluent summary include only the main points outlined in the document. Text summarization can be divided into two categories - **Extractive Summarization** and **Abstractive Summarization**.
- **Extractive Summarization** is based on an extracting several parts, such as phrases and sentences, from a piece of text and stack them together to create a summary. It is important to identifying important phrases or sentences from the original text because it is of utmost importance in this method.
- **Abstractive Summarization** is a relies on generating new sentences from the original text. The sentences generated through this approach might not even be present in the original text. In these methods most often use advanced NLP techniques.

## Project contains:
- text summarization by using several methods - **Text_summary.ipynb**
- text summary by word frequency model with spacy - **spacy_summary.py**
- text summary by TF-IDF model and gensim library - **tfidf_gensim_summary.py**
- text summary by Bart model and transformers - **bart_summary.py**
- data - data used in the project.

## Technologies

**The project is created with:**

- Python  3.6/3.8
- libraries: spacy, heapq, nltk, scikit-learn, gensim, transformers.

**Running the project:**

You can run the scripts in the terminal:

    spacy_summary.py
    tfidf_gensim_summary.py
    bart_summary.py

To run **Text_summary.ipynb** file use Jupyter Notebook or Google Colab.
