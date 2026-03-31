#%%

from typing import Any

import h5py
import numpy as np
import pandas as pd


from pandas import DataFrame

from custom_importer import Importer






global w_options


def main():
    #%% importing the data from out files
    importer1 = Importer()
    dir_location = "cern_data" # location of the directory to search for datafiles
    df = importer1.import_simulation_data_from_directory(dir_location)
    df = importer1.select_test_data_from_dataframe(df)



    # %% select training dataset

    training_data = df[df["is_test_data"] == False]

    X, Y = importer1.build_X_matrix_and_y_vector(training_data)

    #%% run the pocket algorithm


    def hypothesis(x,w):
        return np.sign(np.matmul(x,w))


    #%%

    rng = np.random.default_rng(seed=2869393660352049050)

    #%%setting up w
    # w = pd.DataFrame(data=rng.random([X.shape[1]]),index = X.columns)   # copying the labels from
    # w.iloc[:] = list() # initialize w to random values
    w = rng.random([X.shape[1]])

    #%%
    number_of_iterations = 1000
    recalculate_and_save_every_nr_of_datapoint_points = 10
    w_options = np.zeros([int(number_of_iterations/recalculate_and_save_every_nr_of_datapoint_points),X.shape[1]+1])
    h = hypothesis(np.array(X), w)
    # h, Y = h.align(Y, axis=1, copy=False)
    wrong_hypothesis = h != Y["y"]


#%% computing forloop
    for interation in range(number_of_iterations):

        miscategorized_point_index = X.index[wrong_hypothesis]
        #%%
        point_index = rng.choice(miscategorized_point_index) #pick miscategorized point

        w = w + np.array(Y.loc[point_index, "y"]*X.loc[point_index,:]) # tune weights




        if interation%recalculate_and_save_every_nr_of_datapoint_points==0:
            h = hypothesis(X, w)
            # h, Y = h.align(Y, axis=1, copy=False)
            wrong_hypothesis = h != Y["y"]
            Error_in_sample = 1/len(wrong_hypothesis) * np.sum(wrong_hypothesis)


            # w_options.append((w,Error_in_sample))
            w_options[int(interation/recalculate_and_save_every_nr_of_datapoint_points),:-1] = w
            w_options[int(interation/recalculate_and_save_every_nr_of_datapoint_points),-1] = Error_in_sample
            # w = w + Y*X

    return w_options


    print("end of program")





if __name__ == "__main__":
    #%%

    W_s = main()
    # import cProfile
    #
    # cProfile.run('main()', 'restats2')
    #
