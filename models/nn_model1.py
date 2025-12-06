# New Neural Network model for FEA dataset

#Dec 2025
#************************************************************************************

from lib.args import args
from lib.imports import *
from lib.model_template import ModelTemplate

class NNModel1(ModelTemplate):
    """Neural Network model for Finite Element Analysis (FEA) data, reproduction from the paper:
'*Real-time prediction of projectile penetration to laminates by training
machine learning models with finite element solver as the trainer*' by **Wadagbalkar, Pushkar** and **G.R. Liu** 2021,
Department of Aerospace Engineering and Engineering Mechanics, University of Cincinnati, Cincinnati, OH 45219, USA.
    """
    
    def __init__(self):
        super().__init__()
        # model definition (Neural Network with 3 hidden layers)
        self.model = Sequential()
        self.model.add(Dense(12, activation=LeakyReLU(), input_shape=(12,)))
        self.model.add(Dense(64, activation= LeakyReLU()))
        self.model.add(Dense(128, activation=LeakyReLU()))
        # self.model.add(Dropout(0.1))
        self.model.add(Dense(256, activation=LeakyReLU()))
        self.model.add(Dense(256, activation=LeakyReLU()))
        # self.model.add(Dropout(0.07))
        self.model.add(Dense(1))
        self.model.compile( loss='mse', optimizer=Nadam(learning_rate=0.005))

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
        monitor=EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=15, verbose=1, mode='auto', restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_delta=0.05, verbose=1)
        self.train(epochs=1000, verbose=2, callbacks=[monitor, lr_scheduler])
    
#************************************************************************************

