import re

import pymorphy2
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as sw
from nltk.tokenize import RegexpTokenizer

morph = pymorphy2.MorphAnalyzer()
stopwords = sw.words('russian')
tokenizer = RegexpTokenizer(r'\w+')

custom_stopwords = ['сколько', 'во-сколько', 'здравствовать', 'здрасте', 'вообще', 'это', 'ещё', 'значит', 'значить',
                    'этмый', 'либо', 'хотя', 'таки', 'кроме', 'просто', 'её', 'сей', 'оно', 'ничто', 'го', 'ой',
                    'сегодня', 'спасибо', 'зеленоград', 'москва', 'пермь',
                    'январь', 'февраль', 'март', 'апрель', 'май', 'июнь', 'июль', 'август', 'сентябрь', 'октябрь',
                    'ноябрь', 'декабрь']


def _tokenizer(s):
    return s.split()


def process_word(word):
    return morph.parse(word)[0].normal_form.lower()


def process_list(list_):
    new_list = []
    for l in list_:
        words = tokenizer.tokenize(l)

        new_words = [process_word(word) for word in words
                     if morph.parse(word)[0].normal_form not in stopwords
                     and not any(char.isdigit() for char in word)
                     and not bool(re.search(r'[a-zA-Z]', word))
                     and morph.parse(word)[0].normal_form.lower() not in custom_stopwords
                     ]
        new_list.append(' '.join(w for w in new_words))
    new_list = [elem for elem in new_list if elem != '']
    return new_list


def match_value_with_dummies(value, data, parameter):
    interesting_columns, result_column, value = [], '', int(value)
    for column in data.columns:
        if parameter in str(column):
            interesting_columns.append(column)
    for column in interesting_columns:
        interesting_value = int(column[column.index('_') + 1:])
        if interesting_value == value:
            result_column = column
            break
    if result_column != '':
        data[result_column] = 1
    return data
