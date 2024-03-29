import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from heapq import nlargest

nlp = spacy.load('en_core_web_sm')
stopwords = set(STOP_WORDS)

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


def get_summary(text):
    """Function to calculate word frequency"""
    # Build an nlp object
    doc = nlp(text)
    
    # Calculate word frequencies
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] += 1
                
    # Calculate maximum frequency
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequency

    # Calculate sentence score 
    sent_tokens = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
  
    length = int(len(sentence_scores) * 0.3)
    summary = nlargest(length, sentence_scores, key=sentence_scores.get)
    final_summary = [word.text for word in summary]
    final_summary = ''.join(final_summary)
    return final_summary


def main():
    text = read_data(PATH)
    summary = get_summary(text)
    print(summary)


if __name__ == '__main__':
    main()
