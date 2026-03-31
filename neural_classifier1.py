#%%
import h5py
import numpy as np
import pandas as pd

from custom_importer import Importer
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    #%% importing the data from out files
    importer1 = Importer()
    dir_location = "cern_data" # location of the directory to search for datafiles
    df = importer1.import_simulation_data_from_directory(dir_location)
    df = importer1.select_test_data_from_dataframe(df)

    training_data = df[df["is_test_data"] == False]

    X, Y = importer1.build_X_matrix_and_y_vector(training_data)


#%%


    classifier1 = MLPClassifier(hidden_layer_sizes = (200,), activation= 'tanh', solver = 'adam', learning_rate = 'constant')
    classifier1.fit(X, Y)
    score = classifier1.score(X, Y)

    print(f"in sample error is {1-score}")

