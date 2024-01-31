import os
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.summarizer import summarize

nltk.download('punkt')

PATH = os.path.join('data', 'text1.txt')


def read_data(filepath):
    """Function to read text data"""
    try:
        with open(filepath) as file:
            text = file.read()
            return text
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def create_tfidf_summary(text):
    """Function to get summary by TF-IDF"""
    tokens = sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words='english')
    tf_idf = vectorizer.fit_transform(tokens)

    sentence_score = [i.sum() / len(i.data) for i in tf_idf]
    avg_sent = sum(sentence_score)/len(sentence_score)

    summary = [tokens[index] for index, score in enumerate(sentence_score) if score > avg_sent]
    output_text = ''.join([str(i) for i in summary])
    return output_text


def gensim_summary(text, word_count=100):
    """Function to get summary by gensim library"""
    summ_text = summarize(text, word_count=word_count)
    return summ_text


def main():
    text = read_data(PATH)
    if text:
        summary = create_tfidf_summary(text)
        summary_two = gensim_summary(text)
        print(f'The summary created by TF IDF method:\n{summary}')
        print('----------------------------------')
        print(f'The summary created by gensim library:\n{summary_two}')

if __name__ == '__main__':
    main()
