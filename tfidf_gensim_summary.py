import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.summarizer import summarize
nltk.download('punkt')

path = '\data\text1.txt'


def read_data(filepath):
    """Function to read text data"""
    with open(filepath) as file:
        text = file.read()
        return text


def create_summary(text):
    """Function to get summary by TF-IDF"""
    # split the tokens
    tokens = sent_tokenize(text)

    # create a tf-idf vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tf_idf = vectorizer.fit_transform(tokens)

    # calculating sentence score
    sent_index = 0
    sent_score = []
    for i in tf_idf:
        score = i.sum()/len(i.data)
        sent_index += 1
        sent_score.append(score)

    # calculating average of sentence scores
    avg_sent = sum(sent_score)/len(sent_score)

    # getting summary
    index = 0
    summary = []
    for i in sent_score:
        if (i > (avg_sent)):
            summary.append(tokens[index])
        index += 1
    output_text = ''
    for i in summary:
        output_text = output_text + str(i)
    return output_text


def gensim_summary(text):
    """Function to get summary by gensim library"""
    summ_text = summarize(text, word_count=100)
    return summ_text


def main():
    text = read_data(path)
    summary = create_summary(text)
    summary_two = gensim_summary(text)
    print(f'The summary created by TF IDF method:\n {summary}')
    print('----------------------------------')
    print(f'The summary created by gensim library:\n {summary_two}')


if __name__ == '__main__':
    main()
