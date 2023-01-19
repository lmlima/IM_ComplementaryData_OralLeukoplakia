# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Script to prepare data to train, validate and test PAD-UFES-20 dataset
"""

import sys

import torchvision

sys.path.insert(0, '../../..')  # including the path to deep-tasks folder
from constants import RAUG_PATH

sys.path.insert(0, RAUG_PATH)
import pandas as pd
import os
from raug.utils.loader import split_k_folder_csv, label_categorical_to_number
import pandas as pd
from pathlib import Path

BASE_PATH = "/home/leandro/Documentos/doutorado/dados/sab-patch/data_train_test/data"
# BASE_PATH = "/home/leandro/Documentos/doutorado/dados/sab-patch/sabpatch_displasia"



# clin_ = ["img_id", "diagnostic", "patient_id", "lesion_id", "biopsed"]
#
# clin_feats = ["smoke", "drink", "background_father", "background_mother", "age", "pesticide", "gender", "skin_cancer_history",
#                  "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick", "region", "diameter_1", "diameter_2",
#                  "itch", "grew", "hurt", "changed", "bleed", "elevation"]
# clin_ = ["path", "diagnostico"]
#
# clin_feats_lesao = ["epitelio_alterado", "localizacao", "tamanho_maior"]
# clin_feats_paciente = ["sexo", "uso_bebida", "uso_cigarro"]
#
# clin_feats = clin_feats_lesao + clin_feats_paciente

label = "diagnostico"
# LEUCOPLASIA
# CARCINOMA DE CÉLULAS ESCAMOSAS
label_modification = {'no_dysplasia': 'Ausente', 'carcinoma': 'Câncer', 'with_dysplasia': 'Presente'}

path_train = Path(BASE_PATH, "train")
path_test = Path(BASE_PATH, "test")

train_dl = torchvision.datasets.ImageFolder(path_train)
test_dl = torchvision.datasets.ImageFolder(path_test)

samples_col = ["path", label]
# samples_df = pd.DataFrame.from_dict({c: list() for c in samples_col})
data = []
for img, lab in train_dl.samples:
    # data_zip = zip(img, train_dl.classes[lab] )
    data_dict = {
        "path": img.removeprefix(BASE_PATH).removeprefix("/"),
        label: train_dl.classes[lab]
    }
    data.append(data_dict)

for img, lab in test_dl.samples:
    # data_zip = zip(img, train_dl.classes[lab] )
    data_dict = {
        "path": img.removeprefix(BASE_PATH).removeprefix("/"),
        label: train_dl.classes[lab]
    }
    data.append(data_dict)

samples_df = pd.DataFrame(data, columns=samples_col)


data = split_k_folder_csv(samples_df, label, save_path=None, k_folder=6, seed_number=8)

data_test = data[data['folder'] == 6]
data_train = data[data['folder'] != 6]
data_test.to_csv(os.path.join(BASE_PATH, "sabpatch_parsed_test.csv"), index=False)
label_categorical_to_number(os.path.join(BASE_PATH, "sabpatch_parsed_test.csv"), label,
                            col_target_number="label_number",
                            save_path=os.path.join(BASE_PATH, "sabpatch_parsed_test.csv"),
                            rename_reorder=label_modification)

data_train = data_train.reset_index(drop=True)
data_train = split_k_folder_csv(data_train, label,
                                save_path=None, k_folder=5, seed_number=8)

label_categorical_to_number(data_train, label, col_target_number="label_number",
                            save_path=os.path.join(BASE_PATH, "sabpatch_parsed_folders.csv"),
                            rename_reorder=label_modification)
