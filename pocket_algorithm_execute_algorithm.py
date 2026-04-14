# %%
import h5py
import numpy as np
import pandas as pd
import os
from custom_importer import custom_input_output
from pocket_algorithm import hypothesis_compute_matrix

run_number = 5
file_location = os.getcwd()  # get current dir
file_location = file_location + "/pocket_algorithm_results"  # add save location
file_name = f"pocket_algorithm_{run_number}"  # filename of file to save/load
importer2 = custom_input_output()

name_results = file_name + "results.h5"
name_xy = file_name + "xy.h5"
name_indices = file_name + "used_indices.h5"

importer1 = custom_input_output()
training_data = importer1.import_alternative_data_version("test_chunk_fixed_filtered.csv")
x, y = importer1.build_X_matrix_and_y_vector(training_data)

W_s = importer2.load_dataframe_from_disk(name_results, file_location)
# xy_df = importer2.load_dataframe_from_disk(name_xy, file_location)
# used_indices_df = importer2.load_dataframe_from_disk(name_indices, file_location)

# %%
# y = xy_df["y"]
# x = xy_df.drop(columns=["y", "In sample error"])

# %%
weights = np.array(W_s.drop("In sample error", axis=1))
best_weight_index = np.argmin(W_s["In sample error"])

best_w = weights[best_weight_index, :]
hypothesis_matrix = hypothesis_compute_matrix(np.array(x), best_w)
normalized_prediction = hypothesis_matrix / np.max(np.abs(hypothesis_matrix))

# %%

from neural_classifier1 import compute_Auc_value
from neural_classifier1 import compute_and_show_ROC_curve

y_true = np.array(y)
y_score = normalized_prediction

# %%

frame = pd.DataFrame()
frame["y_true"] = y_true
frame["y_score"] = y_score

#%%
importer2.append_to_dataframe_on_disk(frame, "pocket_algo_test", "data_for_roc")


#
# auc = compute_Auc_value(y_true, y_score)
# compute_and_show_ROC_curve(y_true, y_score)
