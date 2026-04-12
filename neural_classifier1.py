#%%
import json as js
import numpy as np
import pandas as pd
import sklearn.metrics as met
import matplotlib.pyplot as plt


from custom_importer import custom_input_output
from sklearn.neural_network import MLPClassifier


def compute_and_show_ROC_curve(y_true, y_score):
    """
    Takes a 1D array of true values for y, and a 1D array of prediction probabilities of class = 1, and computes, for a bunch of thresholds, the True Positive Rate, and the True Negative Rate.
    Plot the ROC curve, and returns the True Positive Rate, the True Negative Rate, and the thresholds used.

    :param y_true: 1D array of true values for y
    :param y_score: 1D array of prediction probabilities of class = 1
    :return: fpr, tpr, thresholds
    """
    fpr, tpr, thresholds = met.roc_curve(y_true=y_true, y_score=y_score)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return fpr, tpr, thresholds


def compute_Auc_value(y_true, y_score):
    """
    Takes a 1D array of true values for y, and a 1D array of estimated and computes the corresponding AUC value.

    :param y_true:
    :param y_score:
    :return: AUC value
    """
    AUC_score = met.roc_auc_score(y_true=y_true, y_score=y_score)
    return AUC_score


if __name__ == "__main__":
    #%% importing the data from out files

    dir_location = "cern_data" # location of the directory to search for datafiles
    network_params_dir_location = "network_params/"
    location_of_parameter_file = "tanh_200_adam_invscaling_1.json"
    save_parameters = False

    importer1 = custom_input_output()
    df = importer1.import_simulation_data_from_directory(dir_location)
    df = importer1.select_test_data_from_dataframe(df)

    training_data = df[df["is_test_data"] == False]

    X, Y = importer1.build_X_matrix_and_y_vector(training_data)




#%%


    classifier1 = MLPClassifier(activation= 'tanh',hidden_layer_sizes = (200,), solver = 'adam', learning_rate = 'invscaling')
    classifier1.fit(X, Y)
    estimates = classifier1.predict_proba(X)
    score = classifier1.score(X, Y)
    parameters = classifier1.get_params()

    # json_param_dump =
    if save_parameters:
        with open(network_params_dir_location+location_of_parameter_file, "w") as file:
            js.dump(parameters, file, indent=4)


    print(f"in sample error is {1-score}")

#%%

    y_true = Y["y"]
    y_score = estimates[:,1]
    compute_Auc_value(y_true, y_score)
    compute_and_show_ROC_curve(y_true, y_score)


#%%

file_location_neural_network = "network_params/neural_network_1"



# importer1.pickle_file_to_disk(file_location_neural_network,classifier1)
#
# classifier2 = importer1.unpickle_file_to_disk(file_location_neural_network)