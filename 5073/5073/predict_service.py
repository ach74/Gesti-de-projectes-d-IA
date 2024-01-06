
iris_type = {
    0: 'Setosa',
    1: 'Versicolour',
    2: 'Virginica',
}


def logistic_regression(customer, dv, model):
    return predict_with_model(customer, dv, model)

def support_Vector_Machine(customer, dv, model):
    return predict_with_model(customer, dv, model)

def decision_Tree(customer, dv, model):
    return predict_with_model(customer, dv, model)

def knn(customer, dv, model):
    return predict_with_model(customer, dv, model)

def predict_with_model(customer, dv, model):
    length = float(customer['length'])
    width = float(customer['width'])
    
    x = dv.transform([[length, width]])
    data_pred = model.predict(x)[0]

    return iris_type[int(data_pred)],int(data_pred)