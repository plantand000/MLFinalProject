from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

class Model():
    def __init__(self, args):
        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        self.clf = SVC(probability=True)
        self.param_grid = {'C': [10,15,20,21,25,30],
                           'kernel': ['rbf'],
                           'gamma': [0.011, 0.012, 0.013]
                           }
        self.grid_search = GridSearchCV(self.clf, self.param_grid, cv=5)
        # self.pca = PCA(n_components=50)
        ########################################################################

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        # x_train_pca = self.pca.fit_transform(x_train)
        self.grid_search.fit(x_train, y_train)
        print(self.grid_search.best_params_['C'])
        print(self.grid_search.best_params_['kernel'])
        print(self.grid_search.best_params_['gamma'])
        self.clf = self.grid_search.best_estimator_
        self.clf.fit(x_train, y_train)
        return
        ########################################################################

    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x
        # x_pca = self.pca.transform(x)
        preds = self.clf.predict_proba(x)
        preds_auc = self.clf.predict(x)
        return preds,preds_auc
        ########################################################################
        #return preds