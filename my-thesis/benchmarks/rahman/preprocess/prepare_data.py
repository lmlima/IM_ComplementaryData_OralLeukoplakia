# -*- coding: utf-8 -*-
"""

Script to prepare data to train, validate and test Rahman dataset
"""

import sys
sys.path.insert(0,'../../..') # including the path to deep-tasks folder
from constants import RAUG_PATH
sys.path.insert(0,RAUG_PATH)
import os
from raug.utils.loader import create_csv_from_folders, split_k_folder_csv, label_categorical_to_number

BASE_PATH = "/home/leandro/Documentos/doutorado/dados/histo-rahman-20/prepared-rahman20"

data_csv = create_csv_from_folders(F"{BASE_PATH}", img_exts=['jpg'],
                                    save_path=F"{BASE_PATH}/Rahman_full.csv")

data = split_k_folder_csv(data_csv, "target", save_path=None, k_folder=6, seed_number=8)

data_test = data[ data['folder'] == 6]
data_train = data[ data['folder'] != 6]
data_test.to_csv(os.path.join(BASE_PATH, "rahman-20_parsed_test.csv"), index=False)
label_categorical_to_number (os.path.join(BASE_PATH, "rahman-20_parsed_test.csv"), "target",
                             col_target_number="label_number",
                             save_path=os.path.join(BASE_PATH, "rahman-20_parsed_test.csv"))

data_train = data_train.reset_index(drop=True)
data_train = split_k_folder_csv(data_train, "target",
                                save_path=None, k_folder=5, seed_number=8)

label_categorical_to_number (data_train, "target", col_target_number="label_number",
                             save_path=os.path.join(BASE_PATH, "rahman-20_parsed_folders.csv"))
