from transformers import pipeline
from transformers import BartTokenizer, BartForConditionalGeneration

path = '/data/text1.txt'

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


def read_data(filepath):
    """Function to read text data"""
    with open(filepath) as file:
        text = file.read()
        return text


def get_summary(text):
    """Function to get summary by transformers library"""
    summarizer = pipeline("summarization")
    summaries = summarizer(text, min_length=10, max_length=300)
    summary_text = ' '.join([str(i) for i in summaries])
    print(summary_text)


def bart_summary(text):
    """Function to get summary by bart model"""
    input = tokenizer.encode(text, return_tensors="pt", max_length=512)
    summary_ids = model.generate(input, max_length=160, min_length=12,
                                 length_penalty=1.0, num_beams=4,
                                 early_stopping=True)
    output_summ = [tokenizer.decode(g, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
                   for g in summary_ids]
    print(output_summ)


def main():
    text = read_data(path)
    print(f'The summary created by transformers library:\n')
    get_summary(text)
    print('----------------------------------')
    print(f'The summary created by bart model:\n')
    bart_summary(text)


if __name__ == '__main__':
    main()
