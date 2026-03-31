import h5py
import numpy as np
import pandas as pd
import os



class Importer:


    def read_cern_csv_file(self, filename):
        return pd.read_csv(filename, delimiter=' ')

    def assign_label(self, dataframe, label): # labels each dataline with the provided label
        dataframe["label"] = label
        return dataframe

    def import_data_for_use(self, filename1):
        b = self.read_cern_csv_file(filename1)
        self.assign_label(b, filename1.name) #label with filename where data originated
        return b


    def import_simulation_data_from_directory(self,directory_location):
        # scanning through all files in provided dir
        datalist = []
        for file_or_dir in os.scandir(directory_location):
            if file_or_dir.is_file():
                if file_or_dir.__str__().find("data") >= 0:  # if it is measurement data, do nothing
                    pass
                else:
                    datalist.append(self.import_data_for_use(
                        file_or_dir))  # if it is not measurement data, import the data and add it to the list.
        dataframe = pd.concat(datalist,ignore_index=True) #generate dataframe combining all data.
        return dataframe

    def select_test_data_from_dataframe(self, dataframe, seed = 2869393660352049057, test_data_size_fraction=0.1):
        """
        Selects test_data_size_fraction of the entries in dataframe and marks them as either test data or not test data.

        :param dataframe:
        :param seed: optional seed for randomization
        :param test_data_size_fraction: fraction of total data to be used as test data
        :return: dataframe with additional column 'is_test_data' which contains boolean values.
        """
        number_of_entries_in_test_set = int(np.floor(len(dataframe.index)*test_data_size_fraction))
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(dataframe.index)
        dataframe.loc[:, "is_test_data"] = False
        dataframe.loc[shuffled_indices[0:number_of_entries_in_test_set], "is_test_data"] = True
        return dataframe

    def build_X_matrix_and_y_vector(self,df):
        """
        takes the provided dataframe, and labels the Higgs boson production with a 1 in the y column and non Higgs boson production with a -1 in the y column.
        then returns all inputs as a matrix X, and all y outputs as a vector Y.
        :return: X, Y
        """




        # %%
        # identify y for classification
        y_index = []
        y_index = y_index + list(df.index[df["label"] == "ggH125_WW2lep.csv"])
        y_index = y_index + list(df.index[df["label"] == "VBFH125_WW2lep.csv"])

        df.loc[:, "y"] = -1
        df.loc[y_index, "y"] = 1
        Y = pd.DataFrame(df.loc[:, "y"])

        # %%
        # identify X matrix
        X = df.drop(columns=["y", "label", "is_test_data"])

        # %%normalize training data
        normalization = np.max(abs(X), axis=0)  # get max magnitude values of each column
        normalization[normalization == 0] = 1  # remove any division by 0 issues
        X = X / normalization
        return X, Y