import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.summarization.summarizer import summarize
nltk.download('punkt')

path = 'C:\\Users\\PC\\Git_projects\\Text_summary\\data\\text1.txt'


def read_data(filepath):
    with open(filepath) as file:
        text = file.read()
        return text


def create_summary(text):
    tokens = sent_tokenize(text)

    vectorizer = TfidfVectorizer(stop_words='english')
    tf_idf = vectorizer.fit_transform(tokens)

    sent_index = 0
    sent_score = []
    for i in tf_idf:
        score = i.sum()/len(i.data)
        sent_index += 1
        sent_score.append(score)
  
    avg_sent = sum(sent_score)/len(sent_score)

    index = 0
    summary1 = []
    for i in sent_score:
        if (i > (avg_sent)):
            summary1.append(tokens[index])
        index += 1
    output_text = ''
    for i in summary1:
        output_text = output_text + str(i)
    return output_text


def gensim_summary(text):
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
