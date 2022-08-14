from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    if treebank_tag.startswith('V'):
        return wordnet.VERB
    if treebank_tag.startswith('N'):
        return wordnet.NOUN
    if treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_word_list(word_list):
    lemmatizer = WordNetLemmatizer()
    lemmatizered_word_list = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(pos)) for word, pos in pos_tag(word_list)]
    return lemmatizered_word_list
