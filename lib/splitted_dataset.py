from lib.imports import *

class SplittedDatasets:
    def __init__(self, Z, Y, test_size=0.20, val_size=0.50, random_state=415):
        """Split the dataset into training, validation, and testing sets.
        Args:
            Z: Preprocessed input features.
            Y: Target variable.
        Returns:
            train_x, test_x, train_y, test_y, validation_x, validation_y : Split datasets.
        """
        # if Y is not a numpy array
        if not isinstance(Y, np.ndarray):
            Y = Y.to_numpy().astype("float32")
        
        #Splitting the data into training, validation and testing datasets
        train_x, testvalid_x, train_y, testvalid_y = train_test_split(Z, Y, test_size=test_size, random_state=random_state)
        validation_x, test_x, validation_y, test_y = train_test_split(testvalid_x,testvalid_y,test_size=val_size, random_state=random_state)
        
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.test_x = test_x
        self.test_y = test_y
    
    @classmethod
    def manual_split(cls, train_x, test_x, train_y, test_y,
                     validation_x, validation_y):
        """Create a SplittedDatasets instance from pre-split datasets."""
        self = cls.__new__(cls)
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        return self
        
        
    def get_splits(self):
        return (self.train_x, self.test_x, self.train_y, self.test_y,
                self.validation_x, self.test_x, self.validation_y, self.test_y)
        
    def get_testing_set(self):
        return self.test_x, self.test_y
        
    def get_validation_set(self):
        return self.validation_x, self.validation_y
    
    def get_training_set(self):
        return self.train_x, self.train_y