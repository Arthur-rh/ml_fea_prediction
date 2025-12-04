# 1D-CNN Neural Network model for FEA dataset (with interpolation)
# Author : RICHELET Arthur

from lib.args import args
from lib.model_template import ModelTemplate
from lib.imports import *

class CNN1D(ModelTemplate):
    """
    1D Convolutional Neural Network model for predicting the entire 
    velocity–time history from simulation metadata.

    This version groups FEM rows into simulations, interpolates each
    simulation onto a fixed time grid, and feeds sequences into a CNN.
    """

    def __init__(self, time_steps: int = 1000):
        super().__init__()
        self.time_steps = time_steps
        self.n_features = None
        self.model = None

    def _build_model(self):
        """Create the 1D CNN model."""
        
        inputs = Input(shape=(self.time_steps, self.n_features))

        x = Conv1D(64, 3, activation="relu", padding="same")(inputs)
        x = Conv1D(64, 3, activation="relu", padding="same")(x)
        x = MaxPooling1D()(x)

        x = Conv1D(128, 3, activation="relu", padding="same")(x)
        x = GlobalAveragePooling1D()(x)

        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        outputs = Dense(self.time_steps)(x)

        model = Model(inputs, outputs)
        model.compile(
            loss="mse",
            optimizer="adam",
            metrics=["mae"]
        )

        self.model = model
        print(model.summary())
        
    def _fit_global_preprocessors(self, numeric_features: list[int|str]=["Time","Velocity"], categorical_features: list[int|str]=["Projectile","Angle","Laminate"]):

        df = self.X.copy()

        # Fit OneHotEncoder ONCE
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ohe.fit(df[categorical_features])

        # Fit global MinMaxScaler on numeric features
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[numeric_features])

        # Precompute time grid globally
        t_min, t_max = df["Time"].min(), df["Time"].max()
        self.t_fixed = np.linspace(t_min, t_max, self.time_steps)
        
    def preprocess_data(self, categorical_features: list[int|str], numeric_features: list[int|str], max_seq_len: int = 1000):
        self._fit_global_preprocessors(numeric_features, categorical_features)
        grouped = self.X.join(self.Y).groupby(["Velocity","Projectile","Angle","Laminate"])

        X_sequences = []
        Y_sequences = []

        for (vel, proj, ang, lam), group in grouped:
            group = group.sort_values("Time")

            # --- Interpolate target ---
            v_interp = np.interp(self.t_fixed, group["Time"], group[self.Y.name])

            # --- Build metadata table ---
            X_meta = pd.DataFrame({
                "Projectile": [proj]*self.time_steps,
                "Angle": [ang]*self.time_steps,
                "Laminate": [lam]*self.time_steps,
                "Time": self.t_fixed,
                "Velocity": [vel]*self.time_steps
            })

            # --- OneHot encode categorical vars (global fitted) ---
            X_cat = self.ohe.transform(X_meta[categorical_features])

            # --- Scale numeric vars globally ---
            X_num = self.scaler.transform(X_meta[numeric_features])

            # --- Combine ---
            X_full = np.concatenate([X_cat, X_num], axis=1)

            X_sequences.append(X_full)
            Y_sequences.append(v_interp)

        self.X = pad_sequences(X_sequences, maxlen=max_seq_len, dtype="float32", padding="post", truncating="post")
        self.Y = pad_sequences(Y_sequences, maxlen=max_seq_len, dtype="float32", padding="post", truncating="post")
        self.n_features = self.X.shape[2] # Update feature dimension after padding
            
    def average_relative_error(self, y_true, y_pred, eps=1e-8):
        """
        Compute relative error for sequence predictions.
        Returns: (train_error, val_error, test_error)
        """
        error = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))
        return error

    def train_model(self):
        if self.model is None:
            self._build_model()
        
        monitor = EarlyStopping(
            monitor="val_loss",
            patience=30,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        )

        self.train(
            epochs=400,
            batch_size=16,
            verbose=2,
            callbacks=[monitor]
        )
