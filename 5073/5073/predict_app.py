import pickle
from flask import Flask, jsonify, request
from predict_service import *

app = Flask('iris')

@app.route('/regression', methods=['POST'])
def post_data_regression():

    with open('models/1_LogisticRegression.pck', 'rb') as f:
        dv, model = pickle.load(f)

    result = {
        'Typo Iris': logistic_regression(request.get_json(), dv, model),
    }

    return jsonify(result)


@app.route('/svm', methods=['POST'])
def post_data_svm():

    with open('models/2_Support_Vector_Machine.pck', 'rb') as f:
        dv, model = pickle.load(f)

    result = {
        'Typo Iris': support_Vector_Machine(request.get_json(), dv, model),
    }

    return jsonify(result)


@app.route('/tree', methods=['POST'])
def post_data_tree():

    with open('models/3_Decision_Tree.pck', 'rb') as f:
        dv, model = pickle.load(f)

    result = {
        'Typo Iris': decision_Tree(request.get_json(), dv, model),
    }

    return jsonify(result)


@app.route('/KNN', methods=['POST'])
def post_data_KNN():

    with open('models/4_K-NN.pck', 'rb') as f:
        dv, model = pickle.load(f)

    result = {
        'Typo Iris': knn(request.get_json(), dv, model),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=8000)  