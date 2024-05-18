import matplotlib.pyplot as plt 
import numpy as np 
from scipy.optimize import fsolve, minimize, differential_evolution
from pandas import DataFrame, concat
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
import pickle
from PID import *

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    df = DataFrame(data)
    cols, names = [], []
    
    # Input sequence (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(df.shape[1])]
    
    # Forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(df.shape[1])]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(df.shape[1])]
    
    # Concatenate the columns
    agg = concat(cols, axis=1)
    agg.columns = names
    
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg

def get_offline_score(k, model):
	dt = 0.5
	t_sim = np.arange(0,400,dt)

	X_off = np.zeros((t_sim.shape[0], 2))
	U_off = np.zeros((t_sim.shape[0], 2))
	Y_off = np.zeros(t_sim.shape[0])
	Y_pred = np.zeros(t_sim.shape[0])

	X_off[0,:] = [Wa, Wb]
	Y_off[0] = y_f(X_off[0], x0=7.0)
	Y_pred[0] = Y_off[0]
	U_off[:,:] = [u1ss, u2ss]

	U_off[80:, 0] *= 1.25
	U_off[160:, 1] *= 0.75

	U_off[300:, 0] = u1ss*0.60
	U_off[600:, 0] = u1ss
	U_off[500:, 1] = u2ss*1.35

	for n in range(0, t_sim.shape[0]-1):
	    X_off[n+1,:] = x_next(X_off[n], U_off[n], dt)
	    Y_off[n+1] = y_f(X_off[n+1], x0=Y_off[n])

	Y_pred[:k+1] = Y_off[:k+1]
	#print(Y_pred)

	for n in range(k, t_sim.shape[0]-1):
	    data_input = np.column_stack((U_off[n-k:n], Y_pred[n-k:n]))
	    model_input = series_to_supervised(data_input, n_in=k-1).values
	    Y_pred[n+1] = model.predict(model_input)[0]

	score = mse(Y_off,Y_pred)
	return score

def main():
	print("LOADING DATA")
	dataset_treino = np.loadtxt("data/PID_data.csv", delimiter=",")
	dataset_teste = np.loadtxt("data/PID_data_test.csv", delimiter=",")

	t = dataset_treino[:,  0]
	t2 = dataset_teste[:,  0]
	dt = t[1] - t[0]

	dataset_treino = dataset_treino[:, 1:]
	dataset_teste = dataset_teste[:, 1:]

	ks  = []
	models = []
	MSEs_treino = []
	MSEs_teste  = []
	R2_treino = []
	R2_teste = []
	MSE_OFFLINE = []
	print("Training...")
	for k in tqdm(range(1, 11)):
	    df_treino = series_to_supervised(dataset_treino, n_in=k)
	    df_teste = series_to_supervised(dataset_teste, n_in=k)
	    
	    Y = df_treino.iloc[:, -1].values
	    X = df_treino.iloc[:, :-3].values
	    
	    Y2 = df_teste.iloc[:, -1].values
	    X2 = df_teste.iloc[:, :-3].values
	    
	    model = LinearRegression(fit_intercept=True).fit(X,Y)
	    models.append(model)
	    yy1 = model.predict(X)
	    yy2 = model.predict(X2)
	    
	    r21 = model.score(X,Y)
	    r22 = model.score(X2, Y2)

	    ks.append(k)
	    MSEs_treino.append(mse(Y,yy1))
	    MSEs_teste.append(mse(Y2,yy2))
	    R2_treino.append(r21)
	    R2_teste.append(r22)
	    MSE_OFFLINE.append(get_offline_score(k, model))

	print("DONE")
	scores = {
	    	"ks":ks,
	    	"MSE_tr":MSEs_treino,
	    	"MSE_te":MSEs_teste,
	    	"R2 train":R2_treino,
	    	"R2 test":R2_teste,
	    	"MSE_offline":MSE_OFFLINE,
			}
	DataFrame(scores).to_excel("results/LIN/linear_models.xlsx", sheet_name="Linear", index=False)
	#chosen_model = int(input("Digite o Modelo Desejado: "))
	with open("models/LIN_model.pkl", "wb") as f:
		pickle.dump(models[0],f)
	print("MODEL SAVED TO DEVICE")

if __name__ == "__main__":
	main()