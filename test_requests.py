import requests, json

if __name__ == "__main__":
	prediction_data = ["В нашем городе очень плохие дороги. Много выбоин и ям", "У нас в подъезде валяется мусор, он пахнет. Уберите пожалуйста",]
	
	response = requests.get(url='http://localhost:8080/classify', params={'prediction_data': json.dumps(prediction_data), 'predict_proba': json.dumps(True), 'user_mode': json.dumps(True)})
	print(response.json())

	response = requests.get(url='http://localhost:8080/classify', params={'prediction_data': json.dumps(prediction_data), 'predict_proba': json.dumps(False), 'user_mode': json.dumps(False)})
	print(response.json())	
