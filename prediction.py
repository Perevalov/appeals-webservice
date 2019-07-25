import os
import pickle as pkl

import pandas as pd
from keras.models import load_model

from utils import process_list


# Загружаем векторайзер
vectorizer = pkl.load(open(os.path.join('final_models', 'vectorizer.pkl'), 'rb'))

# Загружаем объекты LabelEncoder
category_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'category_label_encoder.pkl'), 'rb'))
executor_label_encoder = pkl.load(
    open(os.path.join('final_models', 'label_encoders', 'executor_label_encoder.pkl'), 'rb'))
theme_label_encoder = pkl.load(open(os.path.join('final_models', 'label_encoders', 'theme_label_encoder.pkl'), 'rb'))

# Загружаем модели
clf_category = pkl.load(open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_category.pkl'), 'rb'))
clf_executor = pkl.load(open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_executor.pkl'), 'rb'))
clf_theme = pkl.load(open(os.path.join('final_models', 'classifiers', 'lvl1', 'clf_theme.pkl'), 'rb'))

def predict(prediction_data, is_predict_proba):
    result = []

    for text in prediction_data:
        tmp = dict()

        prepared_text = ' '.join(process_list(text.split())).strip()
        prepared_text = pd.DataFrame(vectorizer.transform([prepared_text]).toarray())

        if is_predict_proba:
            tmp['category_prediction_proba'] = list(clf_category.predict_proba(prepared_text)[0])
            tmp['executor_prediction_proba'] = list(clf_executor.predict_proba(prepared_text)[0])
            tmp['theme_prediction_proba'] = list(clf_theme.predict_proba(prepared_text)[0])

        clf_category_prediction = clf_category.predict(prepared_text)[0]
        clf_executor_prediction = clf_executor.predict(prepared_text)[0]
        clf_theme_prediction = clf_theme.predict(prepared_text)[0]

        theme_prediction = theme_label_encoder.inverse_transform([clf_theme_prediction])[0]
        executor_prediction = executor_label_encoder.inverse_transform([clf_executor_prediction])[0]
        category_prediction = category_label_encoder.inverse_transform([clf_category_prediction])[0]

        tmp['theme_prediction'] = theme_prediction
        tmp['executor_prediction'] = executor_prediction
        tmp['category_prediction'] = category_prediction

        result.append(tmp)

    return result
