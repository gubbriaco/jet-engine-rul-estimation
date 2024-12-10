import sys
sys.path.append('../../')
from utils import *


def _s_score(targets, predictions):
    diff = predictions - targets
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    
def metrics(targets, predictions):  
    rmse = root_mean_squared_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    s = _s_score(targets, predictions)
    return rmse, mse, mae, r2, s


def get_mean_predictions_per_engine(predictions, windows_generated):
    # predictions per ogni engine 
    # -> divide l'array predictions in sub-array, 
    # ognuno corrispondente alle predizioni di ogni motore
    # np.cumsum calcola la somma cumulativa, cioè gli indici
    # che serviranno per prendere i sub-array
    # se [30, 45, 25, ...], np.cumsum produce [30, 75, 100, ...]
    # allora il primo sub-array avrà 30 valori, il secondo 45, ... 
    # però nell'array originale gli elementi del primo subarray 
    # saranno compresi nel range di indici [0:29], il secondo [30, 74] 
    # e così via 
    # [:-1] perchè dal momento che stiamo prendendo la somma cumulativa, 
    # l'ultimo elemento di tale somma cumulativa non serve perchè servono 
    # solo i punti (gli indici) iniziali dei subarray -> 
    # -> [30, 45, 25], np.cumsum produce [30, 75, 100] ma con [:-1] si
    # prendono solo [30, 75]
    predictions_per_engine = np.split(predictions, np.cumsum(windows_generated)[:-1])
    # media ponderata delle previsioni di RUL per ciascun motore
    mean_predictions_per_engine = [
        np.average(prediction_per_engine, weights = np.repeat(1/window_generated, window_generated)) 
        for prediction_per_engine, window_generated in zip(predictions_per_engine, windows_generated)
    ]
    return mean_predictions_per_engine
