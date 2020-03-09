import pandas as pd
import glob
import os
from FindingEligible import Eligible_V1, Eligible_V2


def Importing(dir, type):
    data = pd.read_sas(dir, format='sas7bdat', encoding='iso-8859-1')
    data.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)


    if type == 'kl':
        data = data[['ID'] + ['SIDE'] + ['V00XRKL']]
        data = data.dropna(how='any', axis=0)
        data = data.reset_index(drop=True)
        data.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kl_multi.csv')
        temp = data['V00XRKL'].copy()
        temp.loc[temp < 2] = 0
        temp.loc[temp > 0] = 1
        data_binary = data.assign(V00XRKL=temp)
        return data_binary
    elif type == 'mr':
        data = data[['ID'] + ['SIDE'] + ['V00MBMSFMA'] + ['V00MBMSFLA'] + ['V00MBMSFMC']
                          + ['V00MBMSFLC'] + ['V00MBMSFLP'] + ['V00MBMSFMP'] + ['V00MBMSSS']
                          + ['V00MBMSTLA'] + ['V00MBMSTLC'] + ['V00MBMSTLP'] + ['V00MBMSTMA']
                          + ['V00MBMSTMC'] + ['V00MBMSTMP']]
        data = data.dropna(how='any', axis=0)
        data = data.reset_index(drop=True)
        temp = data[['ID'] + ['SIDE']].copy()
        data = data.drop(columns=['ID', 'SIDE'])
        data[data > 0] = 1
        data_binary = pd.concat([temp, data], axis=1)

        return data_binary

filename_kl = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kxr_sq_bu00.sas7bdat'
filename_mr = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kmri_sq_moaks_bicl00.sas7bdat'

# mr_BML.columns = ['ID', 'SIDE', 'femur medial anterior', 'femur lateral anterior', 'femur medial central',
#                   'femur lateral central', 'femur lateral posterior', 'femur medial posterior',
#                   'tibia sub-spinous', 'tibia lateral anterior', 'tibia lateral central',
#                   'tibia lateral posterior', 'tibia medial anterior', 'tibia medial central',
#                   'tibia medial posterior']

kl = Importing(filename_kl, type='kl')
bml = Importing(filename_mr, type='mr')

# Importing all contents (.csv files)
# path and directory of the contents
path = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/contents'
directory = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/contents/*.csv'

# storing the names of all contents
filename = [files for root, dirs, files in os.walk(path)]

# transferring contents into data frame
Contents_frames = {}
for i, file in enumerate(glob.glob(directory)):
    Contents_frames[filename[0][i]] = pd.read_csv(file)

# Input the months and MRI sequence here for example: 00m, 12m, 96m for months

''' choose mr sequences from below:
'Bilateral PA Fixed Flexion Knee'
'PA Right Hand'
'AP Pelvis'
'MP_LOCATOR_THIGH'
'AX_T1_THIGH'
'PRESCRITION_THIGH'
'MP_LOCATOR_LEFT'
'COR_IW_TSE_LEFT'
'SAG_3D_DESS_LEFT'
'COR_MPR_LEFT'
'AX_MPR_LEFT'
'SAG_IW_TSE_LEFT'
'MP_LOCATOR_RIGHT'
'COR_IW_TSE_RIGHT'
'SAG_3D_DESS_RIGHT'
'COR_MPR_RIGHT'
'AX_MPR_RIGHT'
'SAG_IW_TSE_RIGHT'
'COR_T1_3D_FLASH_RIGHT'
'SAG_T2_MAP_RIGHT'
'''

months = ['00m']
mri_sequence = ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT', 'Bilateral PA Fixed Flexion Knee']

# Exracting subject ID from base line
subject_baseline = Contents_frames['contents_00m.csv']['ParticipantID']
subject_baseline = subject_baseline.drop_duplicates()
subject_baseline = subject_baseline.reset_index(drop=True)

content00 = Contents_frames['contents_00m.csv']



Final_output_v1 = Eligible_V1(subject_baseline, kl, bml, content00, mri_sequence)
#Final_output_v2 = Eligible_V2(subject_baseline, kl, bml, content00, mri_sequence)


# Final_output_v1.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Final_output_v1.csv')
# Final_output_v2.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Final_output_v2.csv')

# kl.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kl.csv')
# bml.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/bml.csv')

temp = Final_output_v1['ParticipantID'].copy()
temp = temp.drop_duplicates()
temp = temp.reset_index(drop=True)


