import sys
sys.path.append('../')
from utils import *


def cycles2rul(
    cycles:int, 
    upper_rul:int=None
) -> np.ndarray :
    """
    Calculates the Remaining Useful Life (RUL) of an engine based on the number of cycles, 
    with an optional upper limit for the RUL.

    Parameters:
    ----------
    cycles : int
        The total number of engine cycles, representing the lifespan of the engine.
    upper_rul : int, optional
        An optional maximum limit for the RUL. If specified, any RUL value greater than
        `upper_rul` will be capped at this limit. If `upper_rul` is greater than or equal
        to `cycles`, it is ignored.

    Returns:
    -------
    np.ndarray
        A NumPy array representing the RUL for each cycle, starting from `cycles - 1` 
        down to 0. If `upper_rul` is specified and is less than `cycles`, the RUL will 
        be capped at `upper_rul` for the initial cycles.

    Example:
    -------
    >>> cycles2rul(10)
    array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    >>> cycles2rul(10, upper_rul=7)
    array([7, 7, 7, 6, 5, 4, 3, 2, 1, 0])
    
    >>> cycles2rul(10, upper_rul=15)
    array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    """

    # creating a descending array from cycles - 1 to 0
    # start = cycles-1
    # stop  = 0
    # step  = -1
    rul = np.arange(start=(cycles-1), stop=-1, step=-1)

    # applying an upper limit to the RUL if `upper_rul` is specified and is less than `cycles`
    # This limits the initial values in `rul_array` to be no higher than `upper_rul`
    if upper_rul is not None and upper_rul < cycles:
        rul = np.minimum(rul, upper_rul)
        
    return rul


def raw2ts(
    raw_features:np.ndarray, 
    raw_targets:np.ndarray=None, 
    window_length:int=1, 
    shift:int=1
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Generates batches of time-series input and target data from raw_features and raw_targets 
    based on specified window length and shift.

    Parameters:
    ----------
    raw_features : np.ndarray
        2D array of input data with shape (num_samples, num_features).
    raw_targets : np.ndarray, optional
        1D array of target RUL values with shape (num_samples,).
    window_length : int
        Length of each window of data.
    shift : int
        Step size for the moving window. Determines overlap between consecutive batches.

    Returns:
    -------
    ts_features : np.ndarray
        3D array of features organized as time series, with shape (batch_size, window_length, num_features).
    ts_targets : np.ndarray, optional
        1D array of targets corresponding to the end of each time series batch, with shape (batch_size,).
        Returned only if raw_targets is provided.

    Example:
    -------
    >>> # Create sample data with 10 samples and 3 features
    >>> raw_features = np.array([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9],
    ...     [10, 11, 12],
    ...     [13, 14, 15],
    ...     [16, 17, 18],
    ...     [19, 20, 21],
    ...     [22, 23, 24],
    ...     [25, 26, 27],
    ...     [28, 29, 30]
    ... ])
    >>> raw_targets = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    
    >>> # Define window length and shift
    >>> window_length = 3
    >>> shift = 2
    
    >>> # Run function to create batches
    >>> ts_features, ts_targets = raw2ts(raw_features, raw_targets, window_length, shift)
    
    >>> # Display the output
    >>> print("ts_features:\n", ts_features)
    ts_features:
    [[[ 1.  2.  3.]
      [ 4.  5.  6.]
      [ 7.  8.  9.]]
     
     [[ 7.  8.  9.]
      [10. 11. 12.]
      [13. 14. 15.]]
     
     [[13. 14. 15.]
      [16. 17. 18.]
      [19. 20. 21.]]
     
     [[19. 20. 21.]
      [22. 23. 24.]
      [25. 26. 27.]]]
    
    >>> print("ts_targets:\n", ts_targets)
    ts_targets:
    [7. 5. 3. 1.]
    """
    
    batch_size = (len(raw_features) - window_length) // shift + 1
    num_features = raw_features.shape[1]
    
    # initializing an empty 3D array for ts_features with the shape (batch_size, window_length, num_features)
    ts_features = np.empty((batch_size, window_length, num_features))

    if raw_targets is not None:
        ts_targets = np.empty(batch_size)
    
    for batch in range(batch_size):
        # getting the start and end indices for the current window of raw_features
        start_idx = batch * shift
        end_idx = start_idx + window_length
        ts_features[batch] = raw_features[start_idx:end_idx]
        if raw_targets is not None:
            ts_targets[batch] = raw_targets[end_idx - 1]

    if raw_targets is not None:
        return ts_features, ts_targets
    else:
        return ts_features


def raw2tswindows(
    raw_features:np.ndarray, 
    window_length:int, 
    shift:int, 
    num_windows:int=1
) -> tuple[np.ndarray, int]:
    """
    Generates batches of time-series input from raw_features using specified window length, 
    shift, and number of windows.

    Parameters:
    ----------
    raw_features : np.ndarray
        Raw features data, where each row represents a sample and each column a feature.
    window_length : int
        The length of each data window.
    shift : int
        Step size for the moving window, determining the overlap between consecutive batches.
    num_windows : int
        Number of time windows to extract.

    Returns:
    -------
    ts_features : np.ndarray
        Processed features in time-series format.
    num_windows_generated : int
        The actual number of windows generated, which will be `num_windows` if enough data is available,
        or less if data is limited.

    Example:
    -------
    >>> # Create sample raw data with 10 samples and 3 features
    >>> raw_data = np.array([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9],
    ...     [10, 11, 12],
    ...     [13, 14, 15],
    ...     [16, 17, 18],
    ...     [19, 20, 21],
    ...     [22, 23, 24],
    ...     [25, 26, 27],
    ...     [28, 29, 30]
    ... ])

    >>> # Define parameters for the windows
    >>> window_length = 3
    >>> shift = 2
    >>> num_windows = 3

    >>> # Run function to create windows
    >>> ts_features, num_windows_generated = raw2tswindows(raw_data, window_length, shift, num_windows)
    
    >>> # Display the output
    >>> print("ts_features:\n", ts_features)
    ts_features:
    [[[10. 11. 12.]
      [13. 14. 15.]
      [16. 17. 18.]]
     
     [[13. 14. 15.]
      [16. 17. 18.]
      [19. 20. 21.]]
     
     [[16. 17. 18.]
      [19. 20. 21.]
      [22. 23. 24.]]]
    
    >>> print("num_windows_generated:", num_windows_generated)
    num_windows_generated: 3
    """
    
    max_num_batches = (len(raw_features) - window_length) // shift + 1

    # getting the actual number of windows to create, limited by data availability
    num_windows_generated = min(num_windows, max_num_batches)
    # getting the required length of data to create the desired number of batches
    required_len = (num_windows_generated - 1) * shift + window_length

    ts_data_windowed = raw2ts(
        raw_features=raw_features[-required_len:, :],
        raw_targets=None,
        window_length=window_length, 
        shift=shift
    )

    return ts_data_windowed, num_windows_generated
