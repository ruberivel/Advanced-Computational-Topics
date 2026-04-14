#%%
import h5py
import numpy as np
import pandas as pd
import os
from custom_importer import custom_input_output

global w_options


def hypothesis_class_determination(matrix):
    return np.sign(matrix)

def hypothesis_compute_matrix(x, w):
    return np.matmul(x, w)

def main():
    #%% importing the data from out files
    importer1 = custom_input_output()
    dir_location = "cern_data" # location of the directory to search for datafiles
    df = importer1.import_simulation_data_from_directory(dir_location)
    df = importer1.select_test_data_from_dataframe(df)

    # %% select training dataset

    training_data = df[df["is_test_data"] == False]
    training_indices = training_data.index

    X, Y = importer1.build_X_matrix_and_y_vector(training_data)


    #%% run the pocket algorithm

    rng = np.random.default_rng(seed=2869393660352049050)

    #%%setting up w
    # w = pd.DataFrame(data=rng.random([X.shape[1]]),index = X.columns)   # copying the labels from
    # w.iloc[:] = list() # initialize w to random values
    w = rng.random([X.shape[1]])

    #%%
    number_of_iterations = 1000
    recalculate_and_save_every_nr_of_datapoint_points = 100
    w_options = np.zeros([int(number_of_iterations/recalculate_and_save_every_nr_of_datapoint_points),X.shape[1]+1])
    hypothesis_matrix = hypothesis_compute_matrix(np.array(X), w)
    h = hypothesis_class_determination(hypothesis_matrix)
    # h, Y = h.align(Y, axis=1, copy=False)
    wrong_hypothesis = h != Y["y"]
    column_names = list(X.columns)
    column_names.append("In sample error")
#%% computing forloop
    for interation in range(number_of_iterations):

        miscategorized_point_index = X.index[wrong_hypothesis]
        #%%
        point_index = rng.choice(miscategorized_point_index) #pick miscategorized point

        w = w + np.array(Y.loc[point_index, "y"]*X.loc[point_index,:]) # tune weights




        if interation%recalculate_and_save_every_nr_of_datapoint_points==0:
            hypothesis_matrix = hypothesis_compute_matrix(np.array(X), w)
            h = hypothesis_class_determination(hypothesis_matrix)
            wrong_hypothesis = h != Y["y"]
            in_sample_error = 1/len(wrong_hypothesis) * np.sum(wrong_hypothesis)


            # w_options.append((w,Error_in_sample))
            w_options[int(interation/recalculate_and_save_every_nr_of_datapoint_points),:-1] = w
            w_options[int(interation/recalculate_and_save_every_nr_of_datapoint_points),-1] = in_sample_error
            # w = w + Y*X

    return w_options, column_names, X, Y, training_indices


    print("end of program")





if __name__ == "__main__":
    #%%

    W_s, column_names, x,y, used_indices = main()

    # prepare data for storage
    df = pd.DataFrame(W_s, columns=column_names)
    xy_df = pd.DataFrame(x, columns=column_names)
    xy_df["y"] = y
    used_indices_df = pd.DataFrame(used_indices)

    #%%

    run_number = 5
    file_location = os.getcwd() # get current dir
    file_location = file_location + "/pocket_algorithm_results" # add save location
    file_name = f"pocket_algorithm_{run_number}" # filename of file to save/load
    importer2 = custom_input_output()

    name_results = file_name+"results.h5"
    name_xy = file_name+"xy.h5"
    name_indices = file_name+"used_indices.h5"


    #store data
    importer2.append_to_dataframe_on_disk(df,filename=name_results,file_location=file_location)
    importer2.append_to_dataframe_on_disk(xy_df,filename=name_xy,file_location=file_location)
    importer2.append_to_dataframe_on_disk(used_indices_df,filename=name_indices,file_location=file_location)









    #%%
    import h5py
    import numpy as np
    import pandas as pd
    import os
    from custom_importer import custom_input_output


    run_number = 5
    file_location = os.getcwd() # get current dir
    file_location = file_location + "/pocket_algorithm_results" # add save location
    file_name = f"pocket_algorithm_{run_number}" # filename of file to save/load
    importer2 = custom_input_output()

    name_results = file_name+"results.h5"
    name_xy = file_name+"xy.h5"
    name_indices = file_name+"used_indices.h5"

    W_s = importer2.load_dataframe_from_disk(name_results,file_location)
    xy_df = importer2.load_dataframe_from_disk(name_xy,file_location)
    used_indices_df = importer2.load_dataframe_from_disk(name_indices,file_location)




    #%%
    y = xy_df["y"]
    x = xy_df.drop(columns=["y","In sample error"])


#%%
    weights = np.array(W_s.drop("In sample error",axis=1))
    best_weight_index= np.argmin(W_s["In sample error"])


    best_w = weights[best_weight_index,:]
    hypothesis_matrix = hypothesis_compute_matrix(np.array(x), best_w)





