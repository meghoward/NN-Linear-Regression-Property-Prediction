import torch
import pickle
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GridSearchCV
import itertools


class Regressor():

    def __init__(self, x, nb_epoch = 100, hidden_layer_sizes=[64, 64], lr=0.1, batch_size=256, dropout_p = 0.1):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Storing all potential hyperparameters
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epochs = nb_epoch 
        self.batch_size = batch_size
        self.learning_rate = lr
        self.dropout_p = dropout_p
        self.hidden_layer_sizes = hidden_layer_sizes
        return

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x_, y = None, training = False):
        """ 
        Preprocess input data for the neural network.

        This method handles missing values, encodes categorical features, and 
        normalizes the data. If 'training' is True, it also fits the encoder 
        and scaler on the data, which should only be done during the training phase.

        Args:
            x (pd.DataFrame): Input data of shape (batch_size, input_size).
            y (pd.DataFrame, optional): Target data of shape (batch_size, 1). Defaults to None.
            training (bool): Indicates whether the preprocessing is for training or inference.

        Returns:
            torch.tensor: Preprocessed input data.
            torch.tensor or None: Preprocessed target data, or None if y is None.
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Copying the data to avoid modifying the original data - that would cause an error
        x = x_.copy()
        # Filling NaN values
        x.fillna(0, inplace = True)  

        if training:
            # One hot encoding the categorical data (final column) using labelbinarizer
            self.encoder = sklearn.preprocessing.LabelBinarizer()
            ocean_prox = x.iloc[:, -1]
            # Fitting the encoder on the ocean proximity dara
            vectors = pd.DataFrame(self.encoder.fit_transform(ocean_prox.to_numpy()))
            x = x.iloc[:, :-1]
            x = x.reset_index(drop=True)
            vectors = vectors.reset_index(drop=True)
            # Concatenating the encoded data with the rest of the data
            x = pd.concat([x, vectors], axis=1)
            # Renaming the columns - specifically ensuring they are all string values
            x.columns = [str(col) for col in x.columns]
            # Normalising the data using standard scaler
            self.scaler = sklearn.preprocessing.StandardScaler()
            x = self.scaler.fit_transform(x)

            if y is not None:
                # Normalising the target data
                self.y_mean = y.mean().values[0]
                self.y_std = y.std().values[0]
                y = (y - self.y_mean) / self.y_std
                y = torch.tensor(y.values, dtype = torch.float32)
        else:
            ocean_prox = x.iloc[:, -1]
            vectors = pd.DataFrame(self.encoder.transform(ocean_prox.to_numpy()))
            x = x.iloc[:, :-1]
            x = x.reset_index(drop=True)
            x = pd.concat([x, vectors], axis=1)
            x.columns = [str(col) for col in x.columns]
            
            # Normalising the data using standard scaler
            x = self.scaler.transform(x)
            if y is not None:
                # We do not normalise the target data if we are not training, as we will be calculating RMSE score based on comparison with true values
                y = torch.tensor(y.values, dtype = torch.float32)
           
        x = torch.tensor(x, dtype = torch.float32)
        # Return preprocessed x and y, return None for y if it was None
        return x, (y if isinstance(y, torch.Tensor) else None)


        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################
        
    def fit(self, x, y, patience = 10, delta = 0.001):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Preprocessing the data
        X, Y = self._preprocessor(x, y=y, training=True)  # Preprocessing the data
        # Creating the data loader
        dataset = TensorDataset(X, Y)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # Creating the model
        model = Model(self.input_size, self.output_size, hidden_layer_sizes=self.hidden_layer_sizes, dropout_p = self.dropout_p)
        # Loss function and optimizer
        score = nn.MSELoss()
        optimiser = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # Early stopping parameters
        best_loss = float('inf')
        epochs_no_improve = 0

        # Training loops
        for epoch in range(self.nb_epochs):  # num_epochs is the number of epochs you want to train for
            # variable to store total loss for an epoch
            epoch_loss = 0
            for batch, (inputs, targets) in enumerate(data_loader):  # Assuming data_loader is your DataLoader for training data
                # Performing our fitting steps
                outputs = model(inputs)
                loss = score(outputs, targets)
                epoch_loss += loss.item()
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
            # Calculating the average loss over the epoch
            epoch_loss /= len(data_loader)

            # Early stopping - if the loss does not improve by delta for patience epochs, stop training
            if epoch_loss < best_loss - delta:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Printing the loss every 10 epochs
            if epoch % 1 == 0 or epochs_no_improve == patience:
                print(f"Epoch {epoch+1}/{self.nb_epochs}, Loss: {epoch_loss}")
            # Early stopping condition
            if epochs_no_improve == patience:
                print("Early stopping triggered.")
                break
        
        self.model = model
        return self
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x, score = False):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).
        OR  {torch.tensor} -- Predicted value for the given input (batch_size, 1).
        Depending on whether the function is called by self.score or not
        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        if score == False:
            # if new data is being predicted, we need to preprocess it
            X, _ = self._preprocessor(pd.DataFrame(x), training = False) # Do not forget
            y = self.model(X)
            # Denormalising the data
            y = y.detach() * self.y_std + self.y_mean
            # Converting the tensor to numpy array
            y = y.numpy()
        else:
            # Here data is already preprocessed
            y = self.model(x)
            y = y.detach() * self.y_std + self.y_mean
            # This will be when called by self.score, which requires y to be a tensor. We do not need to convert it to numpy array
        return y
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):

        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Preprocessing the data
        X, Y = self._preprocessor(x, y, training = False) # Do not forget
        # Predicting the values
        y_pred = self.predict(X, score = True)
        # Calculating the score
        score = nn.MSELoss()
        mse = score(y_pred, Y)
        # Returning RMSE
        return (mse.item())**0.5

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################



class Model(nn.Module):
        def __init__(self, input_size, output_size, hidden_layer_sizes=[32], dropout_p = 0.5):
            #Build the model
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size
            #self.layers = torch.nn.Linear(self.input_size, 1, dtype = torch.float64)

            # Initialize the module list
            self.layers = nn.ModuleList()

            # Input layer
            self.layers.append(nn.Linear(self.input_size, hidden_layer_sizes[0]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))

            # Hidden layers
            for i in range(len(hidden_layer_sizes) - 1):
                self.layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(dropout_p))

            # Output layer
            self.layers.append(nn.Linear(hidden_layer_sizes[-1], self.output_size))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model

def RegressorHyperParameterSearch(x_train, y_train, x_test, y_test, regressor=None): 
     """
     Performs a hyper-parameter search for fine-tuning the regressor implemented 
     in the Regressor class.

     Arguments:
         x_train: Training data features.
         y_train: Training data targets.
         regressor: An instance of a regressor class compatible with GridSearchCV.
        
     Returns:
         The best fitted model according to the specified hyper-parameters. 
     """

    # Building the grid of parameters we wish to vary
     param_grid = {
         'lr': [0.001, 0.01, 0.1],
         'nb_epoch': [30, 50, 100],
         'hidden_layer_sizes': [[16, 16], [32, 32], [64, 64], [32, 64, 32]],
         'batch_size': [128, 256],
         'dropout_p' : [0.1, 0.3, 0.5],
     }

     # Initializing the grid search criteria 
     best_score = np.inf
     best_params = None

     # Iterate over all combinations of hyperparameters
     for params in itertools.product(*param_grid.values()):
         hyperparams = dict(zip(param_grid.keys(), params))
        
         # Initialize the regressor with current hyperparameters
         regressor = Regressor(x_train, **hyperparams)

         # Train the regressor
         regressor.fit(x_train, y_train)

         # Evaluate the regressor using its built-in score function
         score = regressor.score(x_test, y_test)

         # Update best parameters if current model is better
         if score < best_score:
             best_score = score
             best_params = hyperparams

     print("Best parameters found:", best_params)
     print("Best score found:", best_score)

     # Return the best regressor instance
     best_regressor = Regressor(x_train, **best_params)
     best_regressor.fit(x_train, y_train)
     return best_regressor



            #######################################################################
            #                       ** END OF YOUR CODE **
            #######################################################################


def example_main():


    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    
    regressor.fit(x_train, y_train)
    #save_regressor(regressor)

    prediction = regressor.predict(x_test)
    print("Prediction: ", prediction)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))
    
    # Hyperparameter tuning
    best_regressor = RegressorHyperParameterSearch(x_train, y_train, x_test, y_test)
    print("x_train: ", x_train.shape, type(x_train))
    print("y_train: ", y_train.shape, type(y_train))

    #save_regressor(best_regressor)

    # Evaluate the best model
    error = best_regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))

def load_best():
    regressor = load_regressor()
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))
    


if __name__ == "__main__":
    #example_main()
    load_best()