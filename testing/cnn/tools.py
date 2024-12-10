import sys
sys.path.append('../../')
from utils import *


def _s_score(targets, predictions):
    """
    Computes a custom metric to evaluate the discrepancy between predictions and targets.

    This metric applies an asymmetric penalty to errors:
    - Underestimation errors (predictions < targets) are penalized using the formula: `exp(-diff / 13) - 1`.
    - Overestimation errors (predictions > targets) are penalized using the formula: `exp(diff / 10) - 1`.

    The penalty grows exponentially with the magnitude of the error, with overestimations being penalized 
    more severely than underestimations.

    Args:
        targets (numpy.ndarray): An array of true target values to compare against predictions.
        predictions (numpy.ndarray): An array of predicted values from the model.

    Returns:
        float: The total sum of penalties for all discrepancies between predictions and targets.

    Examples:
        >>> import numpy as np
        >>> targets = np.array([10, 20, 30])
        >>> predictions = np.array([12, 18, 35])
        >>> _s_score(targets, predictions)
        2.718281828459045  # (example value; the actual result depends on the input)

    Notes:
        - This metric is designed for use in scenarios where overestimating the target is considered
          more costly than underestimating it.
        - Since it uses the exponential function, it is highly sensitive to large errors.
    """
    diff = predictions - targets
    return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    
def metrics(targets, predictions):
    """
    Computes multiple evaluation metrics to assess the performance of a predictive model.

    The function calculates the following metrics:
    - **Root Mean Squared Error (RMSE):** Measures the square root of the average squared differences between predictions and targets.
    - **Mean Squared Error (MSE):** Measures the average squared differences between predictions and targets.
    - **Mean Absolute Error (MAE):** Measures the average absolute differences between predictions and targets.
    - **R-squared (R²):** Indicates the proportion of variance in the targets explained by the predictions (coefficient of determination).
    - **Custom S-score:** A custom metric that applies asymmetric penalties to underestimations and overestimations (using `_s_score`).

    Args:
        targets (numpy.ndarray): An array of true target values.
        predictions (numpy.ndarray): An array of predicted values.

    Returns:
        tuple: A tuple containing the computed metrics in the following order:
            - `rmse` (float): Root Mean Squared Error.
            - `mse` (float): Mean Squared Error.
            - `mae` (float): Mean Absolute Error.
            - `r2` (float): R-squared score.
            - `s` (float): Custom S-score.

    Examples:
        >>> import numpy as np
        >>> targets = np.array([10, 20, 30])
        >>> predictions = np.array([12, 18, 35])
        >>> metrics(targets, predictions)
        (2.1602, 4.6667, 3.0, 0.87, 1.2345)  # Example values
    """
    rmse = root_mean_squared_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    s = _s_score(targets, predictions)
    return rmse, mse, mae, r2, s


def get_mean_predictions_per_engine(predictions, windows_generated):
    """
    Computes the weighted mean predictions of Remaining Useful Life (RUL) for each engine 
    based on a list of predictions and the corresponding number of windows generated per engine.

    The function performs the following steps:
    1. **Partition Predictions by Engine:**
       - Splits the `predictions` array into sub-arrays, where each sub-array contains the predictions
         corresponding to a specific engine.
       - The `windows_generated` array specifies how many predictions belong to each engine.
       - `np.cumsum(windows_generated)[:-1]` is used to calculate the indices for splitting, as the cumulative
         sum determines the start and end indices for each engine's predictions.

    2. **Calculate Weighted Mean per Engine:**
       - For each engine, computes the weighted mean of its predictions.
       - Weights are assigned uniformly within each engine, calculated as \(1 / \text{number of windows}\) 
         to ensure equal contribution from each window.

    Args:
        predictions (numpy.ndarray): An array of predictions for all engines combined.
        windows_generated (list or numpy.ndarray): A list or array where each element indicates 
            the number of predictions (or windows) generated for a specific engine.

    Returns:
        list: A list of weighted mean predictions, where each element corresponds to the mean prediction
        for a specific engine.

    Examples:
        >>> import numpy as np
        >>> predictions = np.array([1, 2, 3, 4, 5, 6])
        >>> windows_generated = [2, 4]
        >>> get_mean_predictions_per_engine(predictions, windows_generated)
        [1.5, 4.5]  # Engine 1 (mean of [1, 2]), Engine 2 (mean of [3, 4, 5, 6])
    """
    predictions_per_engine = np.split(predictions, np.cumsum(windows_generated)[:-1])
    # weighted average of RUL predictions for each engine.
    mean_predictions_per_engine = [
        np.average(prediction_per_engine, weights = np.repeat(1/window_generated, window_generated)) 
        for prediction_per_engine, window_generated in zip(predictions_per_engine, windows_generated)
    ]
    return mean_predictions_per_engine
