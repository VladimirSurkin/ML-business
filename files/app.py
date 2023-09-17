import numpy as np
import dill
import pandas as pd
import flask
import glob
import os
import csv


def load_models():
    global models
    models_props = load_models_props()
    for model_path in glob.glob("models/*.dill"):
        with open(model_path, 'rb') as f:
            name = model_path.split('/')[-1].split('.')[0]
            threshold = None
            for m in models_props:
                if m[0] != name:
                    continue
                threshold = float(m[5])

            models[name] = {
                'model': dill.load(f),
                'threshold': threshold
            }


def load_models_props():
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'models_descr'), newline='') as f:
        reader = csv.reader(f)
        return list(reader)


app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static')
dill._dill._reverse_typemap['ClassType'] = type
models = {}


@app.route("/")
def main():
    return flask.render_template("index.html",
                                 user_image=os.path.join(app.config['UPLOAD_FOLDER'], 'roc_curve.jpg'),
                                 models=load_models_props(),
                                 params=params_reqired())
    #return flask.render_template("index.html",
    #                             user_image=os.path.join(app.config['UPLOAD_FOLDER'], 'roc_curve.jpg'),
    #                             models=load_models_props(),
    #                             params=params_reqired())


@app.route("/predict", methods=['GET'])
def predict():
    data = {"success": 0}

    get_params = {}
    model_name = flask.request.args.get('Algo')
    for param_info in params_reqired():
        param = param_info['name']
        if param == 'Algo':
            continue
        val = flask.request.args.get(param)
        get_params.update({param: [val]})

    preds = models[model_name]['model'].predict_proba(pd.DataFrame(get_params))

    data["pred_proba"] = float(preds[:, 1][0])
    data["pred"] = 1 if data["pred_proba"] > models[model_name]['threshold'] else 0
    # data["threshold"] = models[model_name]['threshold']
    data["success"] = 1
    return flask.jsonify(data)


def params_reqired():
    """Словарь из параметров, которые надо передать модели для предскзания"""
    return [  # обязательные параметры
        {'name': 'Algo', 'type': 'str', 'example': 'XGB / LogReg',
         'descr': 'Название лагоритма для предсказания (см. подробности внизу этой страницы)'},
        {'name': 'Geography', 'type': 'str', 'example': 'France / Germany', 'descr': 'Название страны'},
        {'name': 'Gender', 'type': 'str', 'example': 'Female / Male', 'descr': 'Пол'},
        {'name': 'Tenure', 'type': 'str', 'example': '2', 'descr': 'Стаж (кол-во лет в должности)'},
        {'name': 'IsActiveMember', 'type': 'int', 'example': '1 -актив, 0 -нет',
         'descr': 'Является ли клиент активным'},
        {'name': 'CreditScore', 'type': 'int', 'example': '619', 'descr': 'Кредитный рейтинг'},
        {'name': 'Age', 'type': 'int', 'example': '42', 'descr': 'Возраст'},
        {'name': 'Balance', 'type': 'float', 'example': '890.54', 'descr': 'Баланс счёта'},
        {'name': 'NumOfProducts', 'type': 'int', 'example': '1', 'descr': 'Количество заказанных услуг'},
        {'name': 'EstimatedSalary', 'type': 'float', 'example': '101348.88', 'descr': 'Годовая зарплата'},
        {'name': 'HasCrCard', 'type': 'int', 'example': '1 -есть, 0 -нет', 'descr': 'Есть ли кредитная карта'},
    ]


if __name__ == "__main__":
    print('Loading models')

    load_models()

    if not len(models):
        print('No models!')
    else:
        print('Starting web server')
        app.run(host="localhost", port=5000, debug=False)