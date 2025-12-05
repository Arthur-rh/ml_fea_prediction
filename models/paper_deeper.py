# New Neural Network model for FEA dataset

# Author : RICHELET Arthur
# This model is similar to the original PaperModelFEA but with more layers, neurons per
# layer and using Leaky ReLU activation functions to avoid dying ReLU problem.

#************************************************************************************

from lib.args import args
from lib.model_template import ModelTemplate
from lib.imports import *

class DeeperPaperModel(ModelTemplate):
    """Neural Network model for Finite Element Analysis (FEA) data, similar to the PaperModelFEA 
    but with more layers and neurons per layer, and using Leaky ReLU activation functions.
    """
    
    def __init__(self):
        super().__init__()
        # model definition (Neural Network with 3 hidden layers)
        self.model = Sequential()
        self.model.add(Dense(12, input_shape=(12,)))
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(64))
        self.model.add(LeakyReLU())
        self.model.add(Dense(1))
        self.model.compile( loss='mse', optimizer='adam')

    def preprocess_data(self, categorical_features: list[int], numeric_features: list[int]):
        """Preprocess the input features by encoding categorical variables.
        Args:
            X: Input features as a DataFrame.
            Y: Target variable as a Series.
            categorical_features: List of column indices in X that are categorical.
        Returns:
            Nothing. Sets the preprocessed input features and target variable.
        """
        # column Transfer and one hot encoding for categorical features
        ct=ColumnTransformer(transformers=[("oh",OneHotEncoder(),categorical_features)],remainder="passthrough")
        self.X = ct.fit_transform(self.X).astype("float32")

        # feature scaling using MinMaxScaler
        sc = preprocessing.MinMaxScaler(feature_range=(0,1))
        self.X[:,[10,11]] = sc.fit_transform((self.X[:, [10,11]]))
        self.X = pd.DataFrame(self.X)
        self.split_dataset()
        
    def train_model(self):
        # training the model with EarlyStopping callback
        monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=30, verbose=1, mode='auto')
        self.train(epochs=1000, verbose=2, callbacks=[monitor])
    
#************************************************************************************

