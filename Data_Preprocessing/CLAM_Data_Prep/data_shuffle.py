# load required packages
import os
import random
import shutil

clam_dir = 'path/to/where train,val,test data sets need to store/'

# split data into train, val, and test sets
def dataset_shuffle(dataset, path, percent=[0.8, 0.1, 0.1]):
    """
    Input Arg:
        dataset -> path where all tfrecord data stored
        path -> path where you want to save training, testing, and validation data folder
    """

    # return training, validation, and testing path name
    train = path + '/train'
    valid = path + '/valid'
    test = path + '/test'

    # create training, validation, and testing directory only if it is not existed
    if os.path.exists(train) == False:
        os.mkdir(os.path.join(clam_dir, 'train'))
    if os.path.exists(valid) == False:
        os.mkdir(os.path.join(clam_dir, 'valid'))
    if os.path.exists(test) == False:
        os.mkdir(os.path.join(clam_dir, 'test'))

    total_num_data = len(os.listdir(dataset))

    # only shuffle the data when train, validation, and test directory are all empty
    if len(os.listdir(train)) == 0 & len(os.listdir(valid)) == 0 & len(os.listdir(test)) == 0:
        train_names = random.sample(os.listdir(dataset), int(total_num_data * percent[0]))
        for i in train_names:
            train_srcpath = os.path.join(dataset, i)
            shutil.copy(train_srcpath, train)

        valid_names = random.sample(list(set(os.listdir(dataset)) - set(os.listdir(train))),
                                    int(total_num_data * percent[1]))
        for j in valid_names:
            valid_srcpath = os.path.join(dataset, j)
            shutil.copy(valid_srcpath, valid)

        test_names = list(set(os.listdir(dataset)) - set(os.listdir(train)) - set(os.listdir(valid)))
        for k in test_names:
            test_srcpath = os.path.join(dataset, k)
            shutil.copy(test_srcpath, test)