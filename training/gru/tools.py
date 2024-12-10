import sys
sys.path.append('../../')
from utils import *


def _display_training_summary(history):
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    print(history_df.tail().to_string(index=False))

def _plot_training_metrics(history):
    pd.DataFrame(history.history).plot()
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()
    
def display_training_performance(history):
    _display_training_summary(history)
    _plot_training_metrics(history)
    

verbose = 1
batch_size = 128
initial_lr = 0.001
epochs = 100
decay_points = [
    int((epochs * 10) / 100),
    int((epochs * 40) / 100),
]
lr_decay_values = [
    initial_lr,
    (0.1 ** 1) * initial_lr,
    (0.1 ** 2) * initial_lr,
]

def scheduler(epoch):
    if epoch < decay_points[0]:
        return lr_decay_values[0]
    elif decay_points[0] <= epoch < decay_points[1]:
        return lr_decay_values[1]
    else:
        return lr_decay_values[2]

lr_values = [scheduler(epoch) for epoch in range(epochs)]

def plot_lrdecay():
    plt.plot(range(epochs), lr_values, label="Learning Rate", color='orange')
    for point, value in zip(decay_points, lr_decay_values[:-1]):
        plt.plot(point, value, '+', label='Decay at epoch {}'.format(point), color='blue')
        plt.axvline(x=point, color='gray', linestyle='--')
        plt.text(point, value, f"{value:.4f}", color="blue", ha="center", va="bottom", fontsize=10)
    plt.text(epochs-1, lr_decay_values[-1], f"{lr_decay_values[-1]:.5f}", color="blue", ha="right", va="bottom", fontsize=10)
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.title("Step-wise Learning Rate Decay")
    plt.grid(True)
    plt.legend()
    plt.show()


def callbacks(scheduler, verbose, stopping_metric, patience):
    return ([
        keras.callbacks.LearningRateScheduler(scheduler, verbose=verbose),
        keras.callbacks.EarlyStopping(monitor=stopping_metric, patience=patience)
    ])

def loss():
    return keras.losses.mse
    
def optimizer():
    return keras.optimizers.Adam(learning_rate=initial_lr)


class RULEstimator:
    @staticmethod
    def get_model(dataset_name:str, window_length:int, num_features:int, num_targets:int, loss:str, optimizer):
        if dataset_name == datasets_name[0]:
            return FD001Estimator(
                window_length=window_length, 
                num_features=num_features, 
                num_targets=num_targets, 
                loss=loss, 
                optimizer=optimizer
            )
        elif dataset_name == datasets_name[1]:
            return FD002Estimator(
                window_length=window_length, 
                num_features=num_features, 
                num_targets=num_targets, 
                loss=loss, 
                optimizer=optimizer
            )
        elif dataset_name == datasets_name[2]:
            return FD003Estimator(
                window_length=window_length, 
                num_features=num_features, 
                num_targets=num_targets, 
                loss=loss, 
                optimizer=optimizer
            )
        elif dataset_name == datasets_name[3]:
            return FD004Estimator(
                window_length=window_length, 
                num_features=num_features, 
                num_targets=num_targets, 
                loss=loss, 
                optimizer=optimizer
            )
        else:
            raise Exception('{} not valid.'.format(dataset_name))


class FD001Estimator():
    def __init__(self, window_length:int, num_features:int, num_targets:int, loss:str, optimizer):
        self.window_length = window_length
        self.num_features = num_features
        self.num_target = num_targets
        self.input = layers.Input(shape=(self.window_length, self.num_features))
        self.gru1 = layers.GRU(128, return_sequences=True, activation="tanh")
        self.gru2 = layers.GRU(64, return_sequences=True, activation="tanh")
        self.gru3 = layers.GRU(32, activation="tanh")
        self.dense1 = layers.Dense(96, activation="relu")
        self.dense2 = layers.Dense(128, activation="relu")
        self.output = layers.Dense(num_targets)
        self.loss = loss
        self.optimizer = optimizer

    def build(self):
        model = models.Sequential(
            layers=[
                self.input,
                self.gru1,
                self.gru2,
                self.gru3,
                self.dense1,
                self.dense2,
                self.output
            ],
        )
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer
        )
        return model


class FD002Estimator():
    def __init__(self, window_length:int, num_features:int, num_targets:int, loss:str, optimizer):
        self.window_length = window_length
        self.num_features = num_features
        self.num_target = num_targets
        self.input = layers.Input(shape=(self.window_length, self.num_features))
        self.conv1d1 = layers.Conv1D(128, 5, activation = "relu")
        self.conv1d2 = layers.Conv1D(96, 5, activation = "relu")
        self.conv1d3 = layers.Conv1D(32, 5, activation = "relu")
        self.globalavgpool1d = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation = "relu")
        self.dense2 = layers.Dense(128, activation = "relu")
        self.output = layers.Dense(num_targets)
        self.loss = loss
        self.optimizer = optimizer

    def build(self):
        model = models.Sequential(
            layers=[
                self.input,
                self.conv1d1,
                self.conv1d2,
                self.conv1d3,
                self.globalavgpool1d,
                self.dense1,
                self.dense2,
                self.output
            ],
        )
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer
        )
        return model


class FD003Estimator():
    def __init__(self, window_length:int, num_features:int, num_targets:int, loss:str, optimizer):
        self.window_length = window_length
        self.num_features = num_features
        self.num_target = num_targets
        self.input = layers.Input(shape=(self.window_length, self.num_features))
        self.conv1d1 = layers.Conv1D(256, 7, activation = "relu")
        self.conv1d2 = layers.Conv1D(96, 7, activation = "relu")
        self.conv1d3 = layers.Conv1D(32, 7, activation = "relu")
        self.globalavgpool1d = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation = "relu")
        self.dense2 = layers.Dense(128, activation = "relu")
        self.output = layers.Dense(num_targets)
        self.loss = loss
        self.optimizer = optimizer

    def build(self):
        model = models.Sequential(
            layers=[
                self.input,
                self.conv1d1,
                self.conv1d2,
                self.conv1d3,
                self.globalavgpool1d,
                self.dense1,
                self.dense2,
                self.output
            ],
        )
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer
        )
        return model


class FD004Estimator():
    def __init__(self, window_length:int, num_features:int, num_targets:int, loss:str, optimizer):
        self.window_length = window_length
        self.num_features = num_features
        self.num_target = num_targets
        self.input = layers.Input(shape=(self.window_length, self.num_features))
        self.conv1d1 = layers.Conv1D(512, 3, activation = "relu")
        self.conv1d2 = layers.Conv1D(96, 5, activation = "relu")
        self.conv1d3 = layers.Conv1D(32, 5, activation = "relu")
        self.globalavgpool1d = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation = "relu")
        self.dense2 = layers.Dense(128, activation = "relu")
        self.output = layers.Dense(num_targets)
        self.loss = loss
        self.optimizer = optimizer

    def build(self):
        model = models.Sequential(
            layers=[
                self.input,
                self.conv1d1,
                self.conv1d2,
                self.conv1d3,
                self.globalavgpool1d,
                self.dense1,
                self.dense2,
                self.output
            ],
        )
        model.compile(
            loss=self.loss, 
            optimizer=self.optimizer
        )
        return model
