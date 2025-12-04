import os

from models.cnn1d import CNN1D
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress TensorFlow logging

from lib.model_template import ModelTemplate
from lib.imports import *
from lib.args import args

from models.paper_deeper import DeeperPaperModel
from models.paper_model import PaperModelFEA
from models.lstm_gru import LSTM_GRU_Model
    
def run_pipeline(model_instance: ModelTemplate):
    """Run the training and evaluation pipeline for a given model instance.
    Args:
        model_instance (ModelTemplate): An instance of a model derived from ModelTemplate.
    Returns:
        None
    """
    # modified import from single provided file
    dataset = pd.read_csv('Finite_element_entire_data_set.csv')
    # inputs: Time, Velocity, Projectile, Angle, Laminate
    X = dataset.drop(columns=['Residual Velcity'])
    Y = dataset['Residual Velcity']

    # converting target data to numeric type
    Y = pd.to_numeric(Y, errors="raise").astype("float32")
    
    model_instance.load_data(X, Y)
    
    # Preprocessing the data
    categorical_features = ["Projectile", "Angle", "Laminate"]  # indices of categorical features in X
    numeric_features = ["Time", "Velocity"]  # indices of numeric features in X
    
    # Preprocess the data
    model_instance.preprocess_data(categorical_features, numeric_features)
    
    if args.plot_scaling or args.plot_all:
        scaled = model_instance.X.iloc[:,:]
        fig, (ob1,ob2)= plt.subplots(ncols=2 , figsize= (10,10))
        ob1.set_title('Before Scaling')
        sns.kdeplot(dataset['Time'],ax=ob1)
        sns.kdeplot(dataset['Velocity'],ax=ob1)
        ob2.set_title('After Scaling')
        sns.kdeplot(scaled["Time"],ax=ob2)
        sns.kdeplot(scaled["Velocity"],ax=ob2)
        
    model_instance.split_dataset()
    model_instance.train_model()
    if args.save_model: model_instance.save_model()
    
    #Plotting training and validation loss curves
    if args.plot_loss or args.plot_all: model_instance.plot_loss_curves()
    
    #Getting relative errors
    train_error, val_error, test_error = model_instance.relative_errors()
    
    #Calculating regression metrics on validation dataset
    mse_val, mae_val, r2_val = model_instance.regression_metrics(dataset_type='validation')
    
    #Compiling results for printing & saving
    results_string: str = f"""Model Results for {model_instance.get_name()} (hash: {model_instance.hash}):

Training Time: {model_instance.training_time:.4f} seconds
Training Epochs: {model_instance.training_epochs}
Final Training Loss: {model_instance.loss_values[-1]:.4f}
Final Validation Loss: {model_instance.val_loss_values[-1]:.4f}

Relative Error on Validation dataset: {val_error*100:.2f} %
Relative Error on Testing dataset: {test_error*100:.2f} %

Mean Squared Error on Validation dataset: {mse_val:.4f}
Mean Absolute Error on Validation dataset: {mae_val:.4f}
R² Score on Validation dataset: {r2_val:.4f}
"""

    print(results_string)

    if args.dont_save_results:
        return
    
    # open csv file in append mode and write results, if it doesn't exist create it with headers
    file_exists = os.path.isfile(f'results_fea.csv')
    if not file_exists:
        with open(f'results_fea.csv', 'w') as f:
            f.write("model_name,hash,training_time_sec,training_epochs,loss,val_loss,rel_val_error,rel_test_err,val_mse,val_mae,val_r2\n")

    with open(f'results_fea.csv', 'a') as f:
        f.write(f"{model_instance.get_name()},{model_instance.hash},{model_instance.training_time:.4f},{model_instance.training_epochs},{model_instance.loss_values[-1]:.4f},{model_instance.val_loss_values[-1]:.4f},{val_error:.4f},{test_error:.4f},{mse_val:.4f},{mae_val:.4f},{r2_val:.4f}\n")
    
    print(f"Results saved to results_fea.csv\n")
    

if __name__ == "__main__":
    #added gpu support
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices and not args.force_cpu:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Memory growth enabled for GPU")
        except:
            print("Could not set memory growth for GPU")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU
        print("Using CPU for computations")
    
    
    models_list = [
        PaperModelFEA(),
        DeeperPaperModel(),
        CNN1D(),
        LSTM_GRU_Model(use_gru=False),
        LSTM_GRU_Model(use_gru=True)
    ]
    
    if args.model_name == 'list':
        print("\n===== Available models =====")
        for model in models_list:
            print(f"\t- {model.get_name()}")
        print("============================")
        exit(0)
        
    if args.model_name not in [model.get_name() for model in models_list]:
        print(f"Error: Model '{args.model_name}' not found. Use 'list' to see available models.")
        exit(1)
        
    models = {model.get_name(): model for model in models_list}
    print(f"\nSelected Model: {args.model_name}\n")
    if args.run_multiple_times < 1:
        args.run_multiple_times = 1
    else:
        print(f"Running the training {args.run_multiple_times} times:", ) 
    
    for _ in range(args.run_multiple_times):
        run_pipeline(models[args.model_name])
    
    
    
        
    
    
    