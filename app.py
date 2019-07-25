from flask import Flask, render_template, request, flash, session, url_for, redirect
from datetime import timedelta
import requests, json
from flask_cors import CORS

from prediction import predict

app = Flask(__name__)
CORS(app)
app.secret_key = '12345678'

@app.route("/")
def chat():
    return render_template("index.html")

@app.route("/chat")
def about():
    return render_template("chat.html")

@app.route("/classify", methods=['GET'])
def get_bot_response():
	prediction_data = request.args.get('prediction_data')
	if prediction_data: prediction_data = json.loads(prediction_data)
	else: return json.dumps("NO DATA FOUND")

	is_predict_proba = request.args.get('predict_proba')
	if is_predict_proba: is_predict_proba = json.loads(is_predict_proba)
	else: is_predict_proba = False

	is_user_mode = request.args.get('user_mode')
	if is_user_mode: is_user_mode = json.loads(is_user_mode)
	else: is_user_mode = False

	print(is_user_mode)

	if is_user_mode:
		result_dictionary = dict()
		prediction_response = predict(prediction_data, is_predict_proba)
		resp = prediction_response[0]

		category_prediction = resp['category_prediction']
		executor_prediction = resp['executor_prediction']
		theme_prediction = resp['theme_prediction']

		if is_predict_proba:
			result_dictionary['category_prediction_proba'] = resp['category_prediction_proba']
			result_dictionary['executor_prediction_proba'] = resp['executor_prediction_proba']
			result_dictionary['theme_prediction_proba'] = resp['theme_prediction_proba']

		if category_prediction != "":
			response_text = u"Категория: {0}<br> Исполнитель: {1}<br> Ключевые слова: {2}".format(category_prediction.replace('_', ' '), executor_prediction.replace('_', ' '), theme_prediction.replace('_', ' '))
			#print(response_text)
		else:
			response_text = "К сожалению нам не удалось определить категорию вашего обращения :("
			#print(str(response_text))
		
		result_dictionary['response_text'] = response_text

		return json.dumps(result_dictionary)

	result = []
	prediction_response = predict(prediction_data, is_predict_proba)

	for resp in prediction_response:
		tmp = dict()

		tmp['category_prediction'] = resp['category_prediction']
		tmp['executor_prediction'] = resp['executor_prediction']
		tmp['theme_prediction'] = resp['theme_prediction']

		if is_predict_proba:
			tmp['category_prediction_proba'] = resp['category_prediction_proba']
			tmp['executor_prediction_proba'] = resp['executor_prediction_proba']
			tmp['theme_prediction_proba'] = resp['theme_prediction_proba']

		result.append(tmp)

	return json.dumps(result)



if __name__ == "__main__":
    app.run(debug=True, port=8080, host='0.0.0.0')
