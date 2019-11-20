import pandas as pd
import glob
import os

def eligible(months, subject_baseline, kl_xray, bml_mr=None ):
    final_content = pd.DataFrame()



    for id in subject_baseline:
        temp = 0

        for i, month in enumerate(months):
            name_content = 'contents_' + month + '.csv'
            content_specific = Contents_frames[name_content]
            content_person = content_specific.loc[(content_specific['ParticipantID'] == id)]

            if mri_sequence[0] in content_person.values and mri_sequence[1] in content_person.values and mri_sequence[2] in content_person.values:
                temp = temp + 1

        if temp == len(months):
            eligible_id.append(id)
            KL_grade = kl_xray.loc[(kl_xray['ID'] == str(id))]
            final_content = final_content.append(KL_grade)

    return eligible_id, final_content






filename_kl = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kxr_sq_bu00.sas7bdat'
kl_xray = pd.read_sas(filename_kl, format='sas7bdat', encoding='iso-8859-1')

filename_mr = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kmri_sq_moaks_bicl00.sas7bdat'
mri_file = pd.read_sas(filename_mr, format='sas7bdat', encoding='iso-8859-1')

mr_BML = mri_file[['ID']+['SIDE']+['V00MBMSFMA']+['V00MBMSFLA']+['V00MBMSFMC']+['V00MBMSFLC']+['V00MBMSFLP']
                    +['V00MBMSFMP']+['V00MBMSSS']+['V00MBMSTLA']+['V00MBMSTLC']+['V00MBMSTLP']+['V00MBMSTMA']
                    +['V00MBMSTMC']+['V00MBMSTMP']]

mr_BML.columns = ['ID', 'SIDE', 'femur medial anterior', 'femur lateral anterior', 'femur medial central',
                      'femur lateral central', 'femur lateral posterior', 'femur medial posterior',
                      'tibia sub-spinous', 'tibia lateral anterior', 'tibia lateral central',
                      'tibia lateral posterior', 'tibia medial anterior', 'tibia medial central',
                      'tibia medial posterior']



kl_xray = kl_xray[['ID'] + ['SIDE'] + ['V00XRKL']]
kl_xray.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
kl_xray = kl_xray.reset_index(drop=True)
kl_xray = kl_xray.dropna(how='any', axis=0)
kl_xray = kl_xray.reset_index(drop=True)
#nan1 = kl_xray.isna().sum()
kl_temp = kl_xray.copy()
kl_temp = kl_xray['V00XRKL']
kl_temp.loc[kl_temp < 2] = 0
kl_temp.loc[kl_temp > 0] = 1
kl_xray_binary = kl_xray.assign(V00XRKL=kl_temp)
mr_BML.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)

mr_BML = mr_BML.dropna(how='any', axis=0)
mr_BML = mr_BML.reset_index(drop=True)
binary = mr_BML.drop(columns=['ID', 'SIDE'])
binary[binary > 0] = 1
x = mr_BML[['ID']+['SIDE']]
mr_BML_binary = pd.concat([x, binary], axis=1)

#nan2 = mr_BML_binary.isna().sum()
kl_temp = kl_xray.copy()
kl_temp = kl_xray['ID']
kl_temp.drop_duplicates(inplace=True)
kl_temp = kl_temp.reset_index(drop=True)
#ee=kl_temp['SIDE'][1]





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
mri_sequence = ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT','Bilateral PA Fixed Flexion Knee']

# Exracting subject ID from base line
subject_baseline = Contents_frames['contents_00m.csv']['ParticipantID']
subject_baseline = subject_baseline.drop_duplicates()
subject_baseline = subject_baseline.reset_index(drop=True)
eligible_id = []

#eligible_id, final_content = eligible(months, subject_baseline, kl_xray)

#final_content.drop_duplicates(subset=['ID', 'SIDE'], inplace=True)
#final_content = final_content.reset_index(drop=True)

content00 = Contents_frames['contents_00m.csv']
cdsa = content00.loc[(content00['ParticipantID'] == 9000296) & (content00['SeriesDescription']=='Bilateral PA Fixed Flexion Knee')]
#kl_xray_binary.loc[(kl_xray_binary['ID'] == id)

for id in subject_baseline:
    content_person = content00.loc[(content00['ParticipantID'] == id)]

    if mri_sequence[0] in content_person.values and mri_sequence[1] in content_person.values and mri_sequence[
        2] in content_person.values:



        if not kl_xray_binary.loc[(kl_xray_binary['ID'] == str(id)) & (kl_xray_binary['SIDE'] == 1)].empty:

            if not mr_BML_binary.loc[(mr_BML_binary['ID'] == str(id)) & (mr_BML_binary['SIDE'] == 1)].empty:
                p2 = content00.loc[(content00['ParticipantID'] == id) & (
                        content00['SeriesDescription'] == mri_sequence[0])]
                p1 = content00.loc[(content00['ParticipantID'] == id) &
                                   (content00['SeriesDescription'] == mri_sequence[2])]
                eligible_id.append(p1)
                eligible_id.append(p2)

        if not kl_xray_binary.loc[(kl_xray_binary['ID'] == str(id)) & (kl_xray_binary['SIDE'] == 2)].empty:

            if not mr_BML_binary.loc[(mr_BML_binary['ID'] ==str(id)) & (mr_BML_binary['SIDE'] == 2)].empty:
                p3 = content00.loc[(content00['ParticipantID'] == id) & (
                        content00['SeriesDescription'] == mri_sequence[1])]
                p1 = content00.loc[(content00['ParticipantID'] == id) &
                                   (content00['SeriesDescription'] == mri_sequence[2])]
                eligible_id.append(p1)
                eligible_id.append(p3)




final = pd.concat(eligible_id)
d = final[['Folder']+['ParticipantID']+['SeriesDescription']]

d.drop_duplicates(inplace=True)
d.reset_index(drop=True)

d = d.reset_index(drop=True)

