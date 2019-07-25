﻿# Автоматическая классификатор обращений граждан

Авторство задачи: Рязанская область.

## Общее описание логики работы решения
Решение основано на алгоритмов машинного обучения для классификации текстовых данных.
Алгоритмы обучаются размеченных исторических данных, после чего они готовы к автоматической работе с новыми данными (классификация обращений) в реальном времени.
В частности, в сервисе используется несколько моделей, основанных на алгоритме Логистической регрессии. В качестве метода векторизации текста используется TF-IDF.

## Требования к окружению для запуска продукта
Данное решение является кроссплатформенным т.к. оно запускается в виде веб-сервиса на удаленном сервере.

Используемый язык программирования: Python 3.6.

## Сценарий сборки и запуска проекта

Для запуска через Docker необходимо выполнить следующие команды (находясь в корневой папке проекта):

`docker build -t appeals-webservice:latest .
docker run -it -p 8080:8080 appeals-webservice`

Если у вас установлен Python 3.6 и pip, можно запустить проект следующими командами:

`pip install -r requirements.txt python app.py`

В обоих случаях приложение запускается на локальной машине по адресу 0.0.0.0:8080

## Примеры использования
Пример взаимодействия с UI:

Внешний вид главной страницы сервиса
![Главная страница сервиса](https://github.com/Perevalov/appeals-webservice/blob/master/resources/homepage.jpg)

Пример взаимодействия №1
![Пример взаимодействия №1](https://github.com/Perevalov/appeals-webservice/blob/master/resources/example1.jpg)

Пример взаимодействия №2
![Пример взаимодействия №2](https://github.com/Perevalov/appeals-webservice/blob/master/resources/example2.jpg)

Пример взаимодействия с API на Python:

```
import requests, json

if __name__ == "__main__":
	prediction_data = ["В нашем городе очень плохие дороги. Много выбоин и ям", "У нас в подъезде валяется мусор, он пахнет. Уберите пожалуйста",]
	
	response = requests.get(url='http://localhost:8080/classify', params={'prediction_data': json.dumps(prediction_data), 'predict_proba': json.dumps(True), 'user_mode': json.dumps(True)})
	print(response.json())

	response = requests.get(url='http://localhost:8080/classify', params={'prediction_data': json.dumps(prediction_data), 'predict_proba': json.dumps(False), 'user_mode': json.dumps(False)})
	print(response.json())```
