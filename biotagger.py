import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


def extract_and_tag(sentence):
    # Adjust the pattern to extract phrases enclosed in curly braces
    pattern = re.compile(r'\{\|([^|]+)\|\-([A-Z]+)\}')
    matches = pattern.finditer(sentence)
    extracted_terms = []
    clean_sentence = sentence
    offset = 0

    for match in matches:
        phrase = match.group(1)
        tag = match.group(2)
        start = match.start() - offset
        end = start + len(phrase)
        extracted_terms.append((phrase, tag, start, end))

        # Update the clean sentence by removing the tag and braces
        before = clean_sentence[:match.start() - offset]
        after = clean_sentence[match.end() - offset:]
        clean_sentence = before + phrase + after
        offset += len(match.group(0)) - len(phrase)

    return clean_sentence, extracted_terms


def tag_sentence(sentence, extracted_terms):
    words = sentence.split()
    bio_tags = ['O'] * len(words)

    for phrase, tag, start, end in extracted_terms:
        phrase_words = phrase.split()
        current_index = 0

        for i, word in enumerate(words):
            if current_index == start:
                start_index = i
                end_index = start_index + len(phrase_words)
                for j in range(start_index, end_index):
                    bio_tags[j] = f'I-{tag}' if j > start_index else f'B-{tag}'
                break
            current_index += len(word) + 1

    tagged_words = [(word, bio_tag) for word, bio_tag in zip(words, bio_tags)]
    return tagged_words


def get_tagged_sentences(tagged_sentences):
    result = []
    for sentence_id, bio_tags in tagged_sentences:
        for word, tag in bio_tags:
            result.append([sentence_id, word, tag])
    return result


def process_text(text):
    sentences = split_into_sentences(text)
    tagged_sentences = []
    for idx, sentence in enumerate(sentences):
        clean_sentence, extracted_terms = extract_and_tag(sentence)
        bio_tags = tag_sentence(clean_sentence, extracted_terms)
        tagged_sentences.append((idx, bio_tags))
    return get_tagged_sentences(tagged_sentences)


if __name__ == '__main__':
    text1 = "The FA# {|98353|-FA} BUN# {|11235|-BUN} is the {|$882.23|-RENTAMOUNT} details. The DATE project deadline is set for {|July 31s't, 2024.|-RENTD} The client requested an extension to {|August 15, 2024.|-TERM} Please ensure all tasks are completed by {|August 1|-COMMD}"
    result1 = process_text(text1)
    print(result1)
