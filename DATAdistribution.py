import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from FindingEligible import Eligible_V1
import glob
import os
from scipy import stats
import numpy as np


path = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/contents'
directory = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/contents/*.csv'

# storing the names of all contents
filename = [files for root, dirs, files in os.walk(path)]

# transferring contents into data frame
Contents_frames = {}
for i, file in enumerate(glob.glob(directory)):
    Contents_frames[filename[0][i]] = pd.read_csv(file)

def some_function(q, id, bml, kl, lable_v1_kl, lable_v1_mr, mri_sequence):
    if mri_sequence[0] in q.values and mri_sequence[1] in q.values:
        s1 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 1)]
        s2 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 2)]
        s3 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 1)]
        s4 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 2)]
        lable_v1_kl.append(s1)
        lable_v1_kl.append(s2)
        lable_v1_mr.append(s3)
        lable_v1_mr.append(s4)
        return lable_v1_kl, lable_v1_mr

    if mri_sequence[0] in q.values:
        s1 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 1)]
        s2 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 1)]
        lable_v1_kl.append(s1)
        lable_v1_mr.append(s2)
        return lable_v1_kl, lable_v1_mr

    if mri_sequence[1] in q.values:
        s1 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 2)]
        s2 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 2)]
        lable_v1_kl.append(s1)
        lable_v1_mr.append(s2)

        return lable_v1_kl, lable_v1_mr

def DropDuplicates(data):
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)

    return data

def division_a(data, name, val1, val2):
    right = data.loc[data[name] == val1]
    right = DropDuplicates(right)
    left = data.loc[data[name] == val2]
    left = DropDuplicates(left)

    return right, left

def Finding_CorrespondingID(file, target, flag, column1=None, column2=None):

    braket = []
    if flag == 1:


        for i in range(len(file['ID'])):

            temp = target.loc[target['ID'] == str(file['ID'][i])]
            braket.append(temp)

        output = pd.concat(braket)
        output = output.reset_index(drop=True)
        return output

    elif flag == 2:

        for i in range(len(file[column1])):

            temp = target.loc[(target[column1] == file[column1][i]) &
                              (target[column2] == file[column2][i])]
            braket.append(temp)

        output = pd.concat(braket)
        output = output.reset_index(drop=True)
        return output

        return output

def BMLs(file, value, column):

    temp = file[['ID'] + ['SIDE']]
    df = file.drop(columns=column).copy()
    df = df[df != value]
    df = pd.concat([temp, df], axis=1)
    output = df.dropna(how='any', axis=0)
    output = output.reset_index(drop=True)

    return output

def gender(healthy, oa):

    temp = healthy.loc[healthy['P02SEX'] == '1: Male']
    male_hl = temp.reset_index(drop=True)
    temp = healthy.loc[healthy['P02SEX'] == '2: Female']
    female_hl = temp.reset_index(drop=True)

    temp = oa.loc[oa['P02SEX'] == '1: Male']
    male_oa = temp.reset_index(drop=True)
    temp = oa.loc[oa['P02SEX'] == '2: Female']
    female_oa = temp.reset_index(drop=True)

    return female_hl, male_hl, female_oa, male_oa

def plotDistribution(healthy, oa, column):
    plt.figure()

    sns.distplot(healthy[column], rug=True, kde=False, fit=stats.gamma, label="health")
    sns.distplot(oa[column],  rug=True, kde=False, fit=stats.gamma, label="OA")
    plt.legend();


# importing files
# importing eligible subject with their available modalities
filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Final_output_v1.csv'
filename2 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Final_output_v2.csv'
v1 = pd.read_csv(filename1)
v2 = pd.read_csv(filename2)

# importing BML and KL labels
filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kl.csv'
filename2 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/bml.csv'
kl = pd.read_csv(filename1)
bml = pd.read_csv(filename2)

# KL labels multi
filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kl_multi.csv'
kl_multi = pd.read_csv(filename1)

# import information such as BMI and age
filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/OAICompleteData_SAS/allclinical00.sas7bdat'
info = pd.read_sas(filename1, format='sas7bdat', encoding='iso-8859-1')

# import info related to sex of the subjects
filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/MeasInventory.csv'
info_sex = pd.read_csv(filename1)

# desired sequences
mri_sequence = ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT', 'Bilateral PA Fixed Flexion Knee']

# fetch eligible ID without duplicates
v1_id1 = DropDuplicates(v1['ParticipantID'])

lable_v1_mr, lable_v1_kl = [], []

# rearrange labels, outputs = labels for bml and KL in order
for id in v1_id1:
    q = v1.loc[(v1['ParticipantID'] == id)]
    lable_v1_kl, lable_v1_mr = some_function(q, id, bml, kl, lable_v1_kl, lable_v1_mr, mri_sequence)


# converting to Dataframe format
eligible_out_kl = pd.concat(lable_v1_kl)
eligible_out_bml = pd.concat(lable_v1_mr)

# remove duplicates and reset indexes
eligible_out_kl = eligible_out_kl.drop(columns='Unnamed: 0')
eligible_out_bml = eligible_out_bml.drop(columns='Unnamed: 0')
eligible_out_kl = eligible_out_kl.reset_index(drop=True)
eligible_out_bml = eligible_out_bml.reset_index(drop=True)

# dividing  right and left knee
Eligible_right, Eligible_left = division_a(eligible_out_kl, 'SIDE', 1, 2)

# Finding subject with no BML grades in any regions
# columns can be changed depending on which sub-region we are considering

column_medial = ['ID', 'SIDE', 'V00MBMSFMA', 'V00MBMSFMC', 'V00MBMSFMP',
          'V00MBMSSS', 'V00MBMSTMA', 'V00MBMSTMC', 'V00MBMSTMP']

column_lateral = ['ID', 'SIDE', 'V00MBMSFLA', 'V00MBMSFLC', 'V00MBMSFLP',
          'V00MBMSSS', 'V00MBMSTLA', 'V00MBMSTLC', 'V00MBMSTLP']

column_allRegions = ['ID', 'SIDE']

value_noBML = 1
value_BML = 0

no_bml = BMLs(eligible_out_bml, value_noBML, column_allRegions)

no_bml_id = DropDuplicates(no_bml['ID'])

content00 = Contents_frames['contents_00m.csv']

nobmls_OUT = Eligible_V1(no_bml_id, kl, bml, content00, mri_sequence)


# dividing right and left knee from subjects with no BMLs
Eligible_right_no_bml, Eligible_left_no_bml = division_a(no_bml, 'SIDE', 1, 2)

# finding corresponding KL grade (same id which have no BMLs)

kl_no_bml = Finding_CorrespondingID(no_bml, eligible_out_kl, flag=2,
                                    column1='ID', column2='SIDE')



# dividing to 2 part: healthy with no BMLs and OA with no BMLs
hl_nobml, oa_nobml = division_a(kl_no_bml, 'V00XRKL', 0, 1)

# healthy with no BMLs and OA with no BMLs ID
hl_nobml_ID = DropDuplicates(hl_nobml[['ID']+['SIDE']])
oa_nobml_ID = DropDuplicates(oa_nobml[['ID']+['SIDE']])
oa_nobml_ID.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/oa_nobml_ID.csv')
hl_nobml_ID.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/hl_nobml_ID.csv')
hl_nobml_ID_copy = hl_nobml_ID.copy()
remove_n = 285
drop_indices = np.random.choice(hl_nobml_ID.index, remove_n, replace=False)
balance_data_healthy_250 = hl_nobml_ID_copy.drop(drop_indices)
balance_data_healthy_250 = DropDuplicates(balance_data_healthy_250)



kl_multi_nombl = Finding_CorrespondingID(no_bml, kl_multi, flag=2,
                                    column1='ID', column2='SIDE')
kl_multi_nombl = kl_multi_nombl.drop(columns='Unnamed: 0')


# left and righ healthy, left and right OA
hl_nobml_right, hl_nobml_left = division_a(hl_nobml, 'SIDE', 1, 2)
oa_nobml_right, oa_nobml_left = division_a(oa_nobml, 'SIDE', 1, 2)


# Merging all clinical info into one Dataframe
info = info[['ID'] + ['P01BMI'] + ['V00AGE']]
info_sex = info_sex['P02SEX']
clinical_info = pd.concat([info, info_sex], axis=1)

# Finding corresponding clinical info for any subgroup
info_hl_right = Finding_CorrespondingID(hl_nobml_right, clinical_info, flag=1)
info_oa_right = Finding_CorrespondingID(oa_nobml_right, clinical_info, flag=1)

female_hl_right, _right, female_oa_right, male_oa_right = gender(info_hl_right, info_oa_right)

info_hl_left = Finding_CorrespondingID(hl_nobml_left, clinical_info, flag=1)
info_oa_left = Finding_CorrespondingID(oa_nobml_left, clinical_info, flag=1)

female_hl, male_hl, female_oa, male_oa = gender(info_hl_left, info_oa_left)



# plotDistribution(info_hl_right, info_oa_right, column='V00AGE')
# plotDistribution(info_hl_right, info_oa_right, column='P01BMI')
#
# plotDistribution(info_hl_left, info_oa_left, column='V00AGE')
# plotDistribution(info_hl_left, info_oa_left, column='P01BMI')

import matplotlib.pyplot as plt

# matplotlib histogram
bins = np.arange(6) - 0.5
bins_count = 5
arr = plt.hist(kl_multi_nombl['V00XRKL'], color='blue', edgecolor='black', bins=bins)
for i in range(bins_count):
    plt.text(arr[1][i]+0.3, arr[0][i], str(arr[0][i]))

sns.distplot(kl_multi_nombl['V00XRKL'],  rug=True, kde=False, label="KL")

# nobmls_OUT.to_csv(r'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/nobmls_OUT.csv')