# LSTM-GRU Neural Network model for FEA dataset (with interpolation)
# Author : RICHELET Arthur

from lib.args import args
from lib.model_template import ModelTemplate
from lib.imports import *

class LSTM_GRU_Model(ModelTemplate):
    def __init__(self, time_steps=1000, use_gru=False):
        self.time_steps = time_steps  # number of time steps in the sequence
        self.use_gru = use_gru  # whether to use GRU instead of LSTM
        
        # to get the class name correctly
        if self.use_gru:
            self.name = "GRU_Model"
        else:
            self.name = "LSTM_Model"

    def _build_model(self):
        """
        Build the LSTM/GRU model.
        
        use_gru: If True, it will use GRU layers, else LSTM.
        """
        # Input layer (the sequence of features)
        inputs = Input(shape=(self.time_steps, self.n_features))

        # LSTM or GRU layers
        if self.use_gru:
            x = GRU(64, return_sequences=True)(inputs)  # GRU layer with 64 units
        else:
            x = LSTM(64, return_sequences=True)(inputs)  # LSTM layer with 64 units
        
        # dropout against overfitting
        x = Dropout(0.2)(x)

        if self.use_gru:
            x = GRU(128, return_sequences=True)(x)
        else:
            x = LSTM(128, return_sequences=True)(x)

        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x) 
        x = Dense(128, activation="relu")(x)

        outputs = Dense(self.time_steps)(x)  # Output shape (time_steps,)

        # Compile model
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print(model.summary())
        self.model = model
            
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
        if not hasattr(self, "model") or self.model is None:
            self._build_model()
        
        monitor = EarlyStopping(
            monitor="val_loss",
            patience=30,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        )
        
        self.train(
            epochs=300,
            batch_size=16,
            verbose=2,
            callbacks=[monitor]
        )
