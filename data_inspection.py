

# %%
import h5py
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from custom_importer import custom_input_output

# %%
importer1 = custom_input_output()
dir_location = "cern_data"  # location of the directory to search for datafiles
df = importer1.import_simulation_data_from_directory(dir_location)

# %%
df_normalized = importer1.build_X_matrix_and_y_vector(df, normalize_full_dict=True)

# %%

# labels = set(df_normalized["label"])
# columns = list(df_normalized.columns)
# to_drop_from_columns_to_drop = []
# columns_to_drop = ["y", "label", "is_test_data", "weight"]
# for column in columns_to_drop:
#     if column in columns:
#         columns.remove(column)
#     else:
#         to_drop_from_columns_to_drop.append(column)
# for dropping in to_drop_from_columns_to_drop:
#     columns_to_drop.remove(dropping)
#
# for column in columns:
#     for label in labels:
#         data_to_plot = df_normalized.loc[df_normalized["label"] == label, column]
#         plt.hist(data_to_plot, label=label, alpha=0.5, bins=201)
#     if np.min(df_normalized[column]) < 0:
#         plt.xlim(-1, 1)
#     else:
#         plt.xlim(0, 1)
#     plt.title(f"Distribution of {column}")
#     # plt.legend()
#
#     plt.show()

# %%

labels = set(df_normalized["label"])
columns = list(df_normalized.columns)
to_drop_from_columns_to_drop = []
columns_to_drop = ["y", "label", "is_test_data", "weight"]
for column in columns_to_drop:
    if column in columns:
        columns.remove(column)
    else:
        to_drop_from_columns_to_drop.append(column)
for dropping in to_drop_from_columns_to_drop:
    columns_to_drop.remove(dropping)

for column in columns:
    plt.figure()

    is_higgs = np.logical_or(df_normalized["label"]=="ggH125_WW2lep.csv", df_normalized["label"]== "VBFH125_WW2lep.csv")

    if column in ["goodjet_n","Lepton1_charge","Lepton1_type","Lepton2_charge","Lepton2_type"]:
        data_to_plot = df_normalized.loc[is_higgs, column]
        plt.hist(data_to_plot, alpha=0.5, bins=10, color="orange", label="Higgs product")


        data_to_plot = df_normalized.loc[np.logical_not(is_higgs), column]
        plt.hist(data_to_plot, alpha=0.5, bins=10, color="blue", label="Not Higgs product")
    else:

        if np.min(df_normalized[column]) < 0:
            plt.xlim(-1, np.max(df_normalized[column]))
        else:
            plt.xlim(0, np.max(df_normalized[column]))

        data_to_plot = df_normalized.loc[is_higgs, column]
        plt.hist(data_to_plot, alpha=0.5, bins=201, color="orange", label="Higgs product")

        data_to_plot = df_normalized.loc[np.logical_not(is_higgs), column]
        plt.hist(data_to_plot, alpha=0.5, bins=201, color="blue", label="Not Higgs product")



    plt.legend()
    plt.ylabel("Count of binned events")
    plt.xlabel("Normalized signal feature magnitude")
    plt.title(f"Distribution of {column}")
    plt.legend(["Higgs product","Not Higgs product"])
    plt.savefig(f"images/{column}.png")

    # plt.show()


