import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
import time
from datetime import datetime, timedelta
import ipdb

t1 = time.time()

df = pd.read_excel('molecular_weight_data.xlsx')
df = df.sample(frac=1)
values = df.to_numpy()

entradas = values[:, :-1]
e1, e2, e3 = entradas[:, 0], entradas[:, 1], entradas[:, 2]
saida = values[:, -1]

scaler_e1 = MinMaxScaler(feature_range=(0,1))
scaler_e2 = MinMaxScaler(feature_range=(0,1))
scaler_e3 = MinMaxScaler(feature_range=(0,1))
scaler_s = MinMaxScaler(feature_range=(0,1))
e1_scaled = scaler_e1.fit_transform(e1.reshape(-1, 1))
e2_scaled = scaler_e2.fit_transform(e2.reshape(-1, 1))
e3_scaled = scaler_e3.fit_transform(e3.reshape(-1, 1))
s_scaled = scaler_s.fit_transform(saida.reshape(-1, 1))

Teste = 30 / 100
Validacao = 30 / 100
Treinamento = 40 / 100 
assert(Teste + Validacao + Treinamento == 1)

e1_train_val, e1_test, e2_train_val, e2_test, e3_train_val, e3_test, s_train_val, s_test = train_test_split(e1_scaled, e2_scaled, e3_scaled, s_scaled, test_size=Teste, random_state=0)
e1_train, e1_val, e2_train, e2_val, e3_train, e3_val, s_train, s_val = train_test_split(e1_train_val, e2_train_val, e3_train_val, s_train_val, test_size=Validacao/(Validacao+Treinamento), random_state=0)


def noGrid(v, kFolding = False):
    clf = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(v),
                    activation='tanh',
                    learning_rate='constant',
                    learning_rate_init=0.1,
                    tol = 0.00001,
                    max_iter=5000,
                    batch_size=1,
                    momentum=0,
                    random_state=0)
    print(f'\nNeural Configuration: {v}\n')

    if not kFolding:
        # Treinamento ============================================================================
        xyz_train = np.array((e1_train.flatten('F'), e2_train.flatten('F'), e3_train.flatten('F')))
        clf.fit(np.transpose(xyz_train), np.ravel(s_train))
        predict_train = clf.predict(np.transpose(xyz_train))
        mse = mean_squared_error(s_train, predict_train)
        r_2 = r2_score(s_train, predict_train)
        print(f'Training set:\n\tMSE: {mse}\n\tR^2: {r_2}')

        # Validation ============================================================================
        xyz_val = np.array((e1_val.flatten('F'), e2_val.flatten('F'), e3_val.flatten('F')))
        predict_val = clf.predict(np.transpose(xyz_val))
        mse = mean_squared_error(s_val, predict_val)
        r_2 = r2_score(s_val, predict_val)
        print(f'\nValidation set:\n\tMSE: {mse}\n\tR^2: {r_2}')

        # Test ============================================================================
        
        xyz_test = np.array((e1_test.flatten('F'), e2_test.flatten('F'), e3_test.flatten('F')))
        predict_test = clf.predict(np.transpose(xyz_test))
        mse = mean_squared_error(s_test, predict_test)
        r_2 = r2_score(s_test, predict_test)
        print(f'\nTest set:\n\tMSE: {mse}\n\tR^2: {r_2}')

        #ipdb.set_trace()

        plt.figure(0)
        plt.plot(s_test, predict_test, 'ro', label="Estimated values")
        plt.plot([0, 1], [0, 1], 'b', label="Perfect prediction")
        plt.legend('')
        plt.grid()
        plt.show()
    
    else:
        xyz_train_val = np.array((e1_train_val.flatten('F'), e2_train_val.flatten('F'), e3_train_val.flatten('F')))
        scores_cv = cross_val_score(clf, np.transpose(xyz_train_val), np.ravel(s_train_val), scoring='r2', cv=kFolding)
        mean_score = np.mean(scores_cv)
        print(f'Cross Validation\n\nTraining set:\tMean R^2 score: {mean_score}')
        clf.fit(np.transpose(xyz_train_val), np.ravel(s_train_val))

        xyz = np.array((e1_test.flatten('F'), e2_test.flatten('F'), e3_test.flatten('F')))
        predict = clf.predict(np.transpose(xyz))
        mse = mean_squared_error(s_test, predict)
        r_2 = r2_score(s_test, predict)
        
        print(f'\nTest set estimation:')
        print(f'\tMSE: {mse}')
        print(f'\tR^2: {r_2}')

        plt.figure(0)
        plt.plot(s_test, predict, 'ro')
        plt.plot([0,1], [0,1], 'b')
        plt.show()


def grid():
    param_grid = [
    {   'solver' : ['lbfgs'],
        'activation' : ['tanh'],
        'hidden_layer_sizes' : [(4, 4)],
        'learning_rate_init' : [0.1], 
        'tol' : [0.0001],
        'max_iter' : [5000],
        'batch_size' : [1],
        'learning_rate' : ['constant']
       }
    ]
    clf = GridSearchCV( MLPRegressor(), param_grid, scoring='r2', cv=4)
    xyz_train_val = np.array((e1_train_val.flatten('F'), e2_train_val.flatten('F'), e3_train_val.flatten('F')))
    clf.fit(np.transpose(xyz_train_val), np.ravel(s_train_val))
    print('\n\t[ Grid Search Results ] \n')
    print(f'\tBest parameters: {clf.best_params_}')
    print(f'\tScores: {clf.cv_results_["std_test_score"]}')
    print(f'\tBest model score: {clf.best_score_}\n')


def main():
    #grid() # If grid search is needed
    noGrid(((4, 4)))
    #noGrid(((9, 6, 4)))

    t2 = time.time()
    print(f'\nExecution time: {timedelta(seconds=t2-t1)}\n')

    

if __name__=='__main__':
    main()
