from lib.imports import *
from lib.args import args
from lib.splitted_dataset import SplittedDatasets
class ModelTemplate:
    def __init__(self):
        pass
    
    def get_name(self):
        if hasattr(self, 'name'):
            return self.name
        return self.__class__.__name__
    
    def average_relative_error(self, y_true, y_pred, eps=1e-8):
        """Calculate the relative error between true and predicted values.
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
        Returns:
            Relative error as a numpy array.
        """
        denom = np.where(np.abs(y_true) < eps, eps, y_true)
        return np.mean(np.abs((y_true - y_pred.flatten()) / denom))
    
    def _hash_model(self):
        """Hash the model architecture and weights for unique identification.
        Returns:
            str: MD5 hash of the model.
        """
        md5 = hashlib.md5()
        
        # 1. Hash the model architecture
        arch = self.model.to_json()  # architecture as JSON string
        md5.update(arch.encode("utf-8"))
        
        # 2. Hash all weights
        for w in self.model.get_weights():
            # Ensure deterministic byte order
            arr = np.ascontiguousarray(w)
            md5.update(arr.tobytes())

        return md5.hexdigest()
            
    def load_data(self, X, Y):
        self.X = X
        self.Y = Y
        
    def set_splitted_dataset(self, dataset: SplittedDatasets):
        """Set a pre-split dataset.
        Args:
            dataset: An instance of SplittedDatasets containing pre-split data.
        Returns:
            Nothing. Sets the split_dataset attribute.
        """
        self.dataset = dataset
        
    def save_model(self, model_name=None):
        """Save the trained model to a file.
        Args:
            filename: The name of the file to save the model to.
        """
        if model_name is None:
            model_name = self.get_name()
            
        model_filename = f'{model_name}_{self.hash}.keras'
        self.model.save(model_filename)
        print(f'Trained model saved as {model_filename}')
        
    def plot_loss_curves(self):
        """Plot training and validation loss curves."""
        epochs= range(1,len(self.loss_values)+1)
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10,6))
        plt.suptitle(f'Model: {self.__class__.__name__} | Hash: {self.hash}')
        sns.lineplot(x=epochs, y=self.loss_values, label='Training loss', color='blue')
        sns.lineplot(x=epochs, y=self.val_loss_values, label='Validation loss', color='red')
        if args.plot_loss_log: plt.yscale('log')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def test_model(self):
        # getting predictions for training, validation & testing dataset
        """Get predictions for training, validation, and testing datasets."""
        self.y_train_pred= self.model.predict(self.dataset.train_x)
        self.y_validation_pred= self.model.predict(self.dataset.validation_x)
        self.y_test_pred= self.model.predict(self.dataset.test_x)
        
    def split_dataset(self, test_size=0.20, val_size=0.50, random_state=415):
        """Split the dataset into training, validation, and testing sets.
        Args:
            test_size: Proportion of the dataset to include in the test split.
            val_size: Proportion of the test set to include in the validation split.
            random_state: Random seed for reproducibility.
        Returns:
            Nothing. Sets the split_dataset attribute.
        """
        self.dataset = SplittedDatasets(self.X, self.Y, test_size, val_size, random_state)
    
    def train(self, **kwargs):
        """Train the neural network model.
        Args:
            args: Positional arguments for the fit method.
            kwargs: Keyword arguments for the fit method.
        Returns:
            None. Sets the history attribute with training history.
        """
        X,Y = self.dataset.get_training_set()
        #Computing training and validation loss for the network
        batch_size = kwargs.pop('batch_size', None)
        epochs = kwargs.pop('epochs', 1)
        verbose = kwargs.pop('verbose', 'auto')
        callbacks = kwargs.pop('callbacks', [])
        start_time = time()
        self.history=self.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=self.dataset.get_validation_set(),
            **kwargs
        )
        end_time = time()
        
        history_dict=self.history.history
        self.loss_values=history_dict['loss']
        self.val_loss_values=history_dict['val_loss']
        
        # makes a hash for identifying the model uniquely based on its architecture and weights
        self.hash = self._hash_model()[:6]
        self.training_epochs = len(self.loss_values)
        self.training_time = end_time - start_time
        
    def relative_errors(self):
        """Calculate relative errors for training, validation, and testing datasets.
        Returns:
            (train_error, val_error, test_error): Tuple of relative errors for training, validation, and testing datasets.
        """
        if not hasattr(self, 'y_train_pred') or not hasattr(self, 'y_validation_pred') or not hasattr(self, 'y_test_pred'):
            self.test_model()
        
        train_error = self.average_relative_error(self.dataset.train_y, self.y_train_pred)
        val_error = self.average_relative_error(self.dataset.validation_y, self.y_validation_pred)
        test_error= self.average_relative_error(self.dataset.test_y, self.y_test_pred)
        
        return train_error, val_error, test_error
    
    def regression_metrics(self, dataset_type='validation'):
        """Calculate regression metrics: MSE, MAE, and R2 score.
        Args:
            y_true: True target values.
            y_pred: Predicted target values.
        Returns:
            mse: Mean Squared Error.
            mae: Mean Absolute Error.
            r2: R2 Score.
        """
        # Ensure predictions are available
        if not hasattr(self, 'y_train_pred') or not hasattr(self, 'y_validation_pred') or not hasattr(self, 'y_test_pred'):
            self.test_model()
        
        # Select the appropriate dataset & predictions
        if dataset_type == 'training':
            y_true = self.dataset.train_y
            y_pred = self.y_train_pred
        elif dataset_type == 'testing':
            y_true = self.dataset.test_y
            y_pred = self.y_test_pred
        else:  # default to validation
            y_true = self.dataset.validation_y
            y_pred = self.y_validation_pred
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return mse, mae, r2