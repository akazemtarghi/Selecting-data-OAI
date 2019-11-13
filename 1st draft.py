import pandas as pd
import glob
import os


def Eligible(months, subject_baseline,kl_xray,bml_mr=None ):
    final_content = pd.DataFrame()

    for i, month in enumerate(months):
        name_content = 'contents_' + month + '.csv'
        for id in subject_baseline:
            content_specific = Contents_frames[name_content]
            content_person = content_specific.loc[(content_specific['ParticipantID'] == id)]

            if mri_sequence[0] and mri_sequence[1] in content_person.values:

                eligible_id.append(id)
                KL_grade = kl_xray.loc[(kl_xray['ID'] == str(id))]
                final_content = final_content.append(KL_grade)

            else:
                print('noo')

    return eligible_id, final_content


filename = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Selecting Data/kxr_sq_bu00.sas7bdat'
kl_xray = pd.read_sas(filename, format='sas7bdat', encoding='iso-8859-1')

# filename = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Selecting Data/kmri_sq_moaks_bicl00.sas7bdat'
# bml_mr = pd.read_sas(filename, format='sas7bdat', encoding='iso-8859-1')

kl_xray = kl_xray[['ID'] + ['SIDE'] + ['V00XRKL']]
kl_xray.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
kl_xray = kl_xray.reset_index(drop=True)





# Importing all contents (.csv files)

# path and directory of the contents
path= 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/contents'
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
'PRESCRIPTION_THIGH'
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
mri_sequence = ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT']

# Exracting subject ID from base line
subject_baseline = Contents_frames['contents_00m.csv']['ParticipantID']
subject_baseline = subject_baseline.drop_duplicates()
subject_baseline = subject_baseline.reset_index(drop=True)
eligible_id = []

eligible_id, final_content = Eligible(months, subject_baseline,kl_xray)

final_content.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
final_content = final_content.reset_index(drop=True)





