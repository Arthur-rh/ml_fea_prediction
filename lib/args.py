import argparse
# Argument parser for command line options
parser = argparse.ArgumentParser(description="Train a neural network model for FEA data")

# arguement to select the model architecture
parser.add_argument(
    "model_name",
    type=str,
    help="Name of the model architecture to use, use 'list' to see available models"
)
# arguement to save the trained model in order to make future predictions without retraining
parser.add_argument(
    "--save-model",
    action="store_true",
    help="Save trained model after training"
)
# arguement to plot training loss curves after training
parser.add_argument(
    "--plot-loss",
    action="store_true",
    help="Plot training loss after training"
)
# arguement to plot training loss curves on log scale after training
parser.add_argument(
    "--plot-loss-log",
    action="store_true",
    help="Plot training loss on a logy scale after training"
)
# argument to plot feature scaling results
parser.add_argument(
    "--plot-scaling",
    action="store_true",
    help="Plot feature scaling results"
)
# arguement to plot all available plots
parser.add_argument(
    "--plot-all",
    action="store_true",
    help="Plot all available plots"
)
# argument to force CPU usage even if GPU is available
parser.add_argument(
    "--force-cpu",
    action="store_true",
    help="Force the use of CPU even if GPU is available"
)
parser.add_argument(
    "--dont-save-results",
    action="store_true",
    help="Do not save results to CSV file"
)
parser.add_argument(
    "--run-multiple-times",
    type=int,
    default=1,
    help="Run the training multiple times"
)
args = parser.parse_args()