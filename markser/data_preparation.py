import os
import shutil

# directory of project
BASE_DIR = r"/home/andrei/work/t7_cv/trafic_signs/markser_data/"

DATA_DIR = os.path.join(BASE_DIR, r"data/train")
VAL_DIR = os.path.join(BASE_DIR, r"data/validation")

def split_data(percent):
    """Move some percent of data from train directory to validation directory 

    # Arguments
        percent: how much samples will be moved to validation directory
        e.g. 0.25
    """
    os.chdir(DATA_DIR)

    if not os.path.exists(VAL_DIR):
        os.mkdir(VAL_DIR)

    for i in os.listdir():

        val_class_dir = os.path.join(VAL_DIR, i)
        data_class_dir = os.path.join(DATA_DIR, i)

        if not os.path.exists(val_class_dir):
            os.mkdir(val_class_dir)

        if os.path.isdir(i):
            directory_list = os.listdir(data_class_dir)
            for j in range(0, round(len(directory_list) * percent)):
                source = os.path.join(data_class_dir, directory_list[j])
                destination = os.path.join(val_class_dir, directory_list[j])
                shutil.move(source, destination)
    
    os.chdir(BASE_DIR)

def number_of_samples(data_from):
    """Indicate amount of samples in train or validation directories

    # Arguments
        data_from: 
        - `train` to count samples from train directory
        - `valid` to count samples from validation directory
    """

    if data_from == "train":
        working_dir = DATA_DIR
    elif data_from == "valid":
        working_dir = VAL_DIR
        
    os.chdir(working_dir)
    
    sum = 0
    for i in os.listdir():

        data_class_dir = os.path.join(working_dir, i)

        if os.path.isdir(data_class_dir):
            sum += len(os.listdir(data_class_dir))
            
    os.chdir(BASE_DIR)
        
    return sum
