
import pandas as pd

from neural_classifier1 import compute_Auc_value
from neural_classifier1 import compute_and_show_ROC_curve
from custom_importer import custom_input_output


importer1 = custom_input_output()
training_data = importer1.import_alternative_data_version("test_chunk_fixed_filtered.csv")
X, Y = importer1.build_X_matrix_and_y_vector(training_data)

import os

run_number = 2
# file_location_neural_network = f"network_params/neural_network_{run_number}"
file_location = os.getcwd()  # get current dir
file_location = file_location + "/network_params"  # add save location
file_name = file_location + f"neural_network{run_number}_"  # filename of file to save/load

name_results = file_name + "results"



classifier1 = importer1.unpickle_file_to_disk(name_results)



estimates = classifier1.predict_proba(X)
score = classifier1.score(X, Y)
parameters = classifier1.get_params()

print(f"in sample error is {1 - score}")

y_true = Y["y"]
y_score = estimates[: ,1]
AUC_score = compute_Auc_value(y_true, y_score)
compute_and_show_ROC_curve(y_true, y_score)

print(AUC_score)


frame = pd.DataFrame()
frame["y_true"] = y_true
frame["y_score"] = y_score

#%%
importer1.append_to_dataframe_on_disk(frame, "neural_classifier_test", "data_for_roc")




# column_names = list(X.columns)
# xy_df = pd.DataFrame(X, columns=column_names)
# xy_df["y"] = y_true

