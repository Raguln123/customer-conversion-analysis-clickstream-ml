from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

def train_classification_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    joblib.dump(clf, '../models/classification_model.pkl')
    return clf

def train_regression_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    print('RMSE:', mean_squared_error(y_test, preds, squared=False))
    joblib.dump(reg, '../models/regression_model.pkl')
    return reg
