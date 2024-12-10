import sys
sys.path.append('../../')
from utils import *


epochs = 100
max_epochs = 20
factor = 3

decay_points = [
    int((epochs * 10) / 100),
    int((epochs * 40) / 100),
]

def scheduler_tuning(epoch, initial_lr):
    if epoch < decay_points[0]:
        return initial_lr
    elif decay_points[0] <= epoch < decay_points[1]:
        return (0.1 ** 1) * initial_lr
    else:
        return (0.1 ** 2) * initial_lr


class RULEstimator_HyperModel(kt.HyperModel):
    def __init__(
        self, 
        window_length:int, 
        num_features:int, 
        num_targets:int, 
        num_conv1d_layers_bounds:Union[List[int], np.ndarray], 
        num_conv1d_filters_bounds:Union[List[int], np.ndarray], 
        num_dense_layers_bounds:Union[List[int], np.ndarray],
        num_dense_units_bounds:Union[List[int], np.ndarray],
        initial_lr_bounds:Union[List[int], np.ndarray],
        batch_size_bounds:Union[List[int], np.ndarray]
    ):
        assert len(num_conv1d_layers_bounds) == 2, "num_conv1d_layers_bounds must have [min, max]."
        assert len(num_conv1d_filters_bounds) == 3, "num_conv1d_filters_bounds must have [min, max, step]."
        assert len(num_dense_layers_bounds) == 2, "num_conv1d_layers_bounds must have [min, max]."
        assert len(num_dense_units_bounds) == 3, "num_dense_units_bounds must have [min, max, step]."
        assert len(initial_lr_bounds) == 2, "initial_lr_bounds must have [min, max]."
        assert len(batch_size_bounds) == 3, "batch_size_bounds must have [min, max, step]."
        
        self.window_length = window_length
        self.num_features = num_features
        self.num_targets = num_targets
        
        self.num_conv1d_layers_bounds = num_conv1d_layers_bounds
        self.num_conv1d_layers_lowerb = self.num_conv1d_layers_bounds[0]
        self.num_conv1d_layers_upperb = self.num_conv1d_layers_bounds[1]
        
        self.num_conv1d_filters_bounds = num_conv1d_filters_bounds
        self.num_conv1d_filters_lowerb = self.num_conv1d_filters_bounds[0]
        self.num_conv1d_filters_upperb = self.num_conv1d_filters_bounds[1]
        self.num_conv1d_filters_step = self.num_conv1d_filters_bounds[2]
        
        self.num_dense_layers_bounds = num_dense_layers_bounds
        self.num_dense_layers_lowerb = self.num_dense_layers_bounds[0]
        self.num_dense_layers_upperb = self.num_dense_layers_bounds[1]
        
        self.num_dense_units_bounds = num_dense_units_bounds
        self.num_dense_units_lowerb = self.num_dense_units_bounds[0]
        self.num_dense_units_upperb = self.num_dense_units_bounds[1]
        self.num_dense_units_step = self.num_dense_units_bounds[2]
        
        self.initial_lr_bounds = initial_lr_bounds
        self.initial_lr_lowerb = self.initial_lr_bounds[0]
        self.initial_lr_upperb = self.initial_lr_bounds[1]
        
        self.batch_size_bounds = batch_size_bounds
        self.batch_size_lowerb = self.batch_size_bounds[0]
        self.batch_size_upperb = self.batch_size_bounds[1]
        self.batch_size_step = self.batch_size_bounds[2]
        

    def build(self, hp):
        # Input Layer
        input_layer = layers.Input(shape=(self.window_length, self.num_features))
        
        # tuning Conv1D Layers
        x = input_layer
        for i in range(hp.Int(
            'num_conv_layers', 
            self.num_conv1d_layers_lowerb, 
            self.num_conv1d_layers_upperb
        )):
            x = layers.Conv1D(
                filters=hp.Int(
                    f'conv_filters_{i}', 
                    min_value=self.num_conv1d_filters_lowerb, 
                    max_value=self.num_conv1d_filters_upperb, 
                    step=self.num_conv1d_filters_step
                ),
                kernel_size=hp.Int(f'conv_kernel_size_{i}', min_value=3, max_value=7, step=2),
                activation='relu'
            )(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # tuning Dense Layers
        for i in range(hp.Int(
            'num_dense_layers', 
            self.num_dense_layers_lowerb, 
            self.num_dense_layers_upperb
        )):
            x = layers.Dense(
                units=hp.Int(
                    f'dense_units_{i}', 
                    min_value=self.num_dense_units_lowerb, 
                    max_value=self.num_dense_units_upperb, 
                    step=self.num_dense_units_step
                ),
                activation='relu'
            )(x)
        
        # Output Layer
        output_layer = layers.Dense(self.num_targets)(x)

        # building the model
        model = models.Model(inputs=input_layer, outputs=output_layer)
        
        # Initial Learning Rate
        initial_lr = hp.Float(
            'initial_lr', 
            min_value=self.initial_lr_lowerb, 
            max_value=self.initial_lr_upperb, 
            sampling='log'
        )

        optimizer = keras.optimizers.Adam(learning_rate=initial_lr)

        # compiling the model
        model.compile(
            optimizer=optimizer,
            loss=losses.mse
        )
        return model
        

    def fit(self, hp, model, *args, **kwargs):
        initial_lr = hp.get('initial_lr')
        user_callbacks = kwargs.pop('callbacks', [])
        lr_scheduler = callbacks.LearningRateScheduler(
            lambda epoch: scheduler_tuning(epoch, initial_lr), verbose=verbose
        )
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
        all_callbacks = [lr_scheduler, early_stopping] + user_callbacks
        return model.fit(
            *args,
            batch_size=hp.Int(
                'batch_size', 
                min_value=self.batch_size_lowerb, 
                max_value=self.batch_size_upperb, 
                step=self.batch_size_step
            ),
            verbose=verbose,
            callbacks=[all_callbacks],
            **kwargs,
        )
