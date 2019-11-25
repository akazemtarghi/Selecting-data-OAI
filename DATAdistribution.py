import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def Corresponding(file, target):

    braket = []
    for i in range(len(file['ID'])):
        temp = target.loc[target['ID'] == str(file['ID'][i])]
        braket.append(temp)

    output = pd.concat(braket)
    output = output.reset_index(drop=True)
    return output



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

# import information such as BMI and age
filename = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/OAICompleteData_SAS/allclinical00.sas7bdat'
info = pd.read_sas(filename, format='sas7bdat', encoding='iso-8859-1')

# import info related to sex of subjects
filename = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/MeasInventory.csv'
info_sex = pd.read_csv(filename)

# desired sequences
mri_sequence = ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT', 'Bilateral PA Fixed Flexion Knee']

# fetch eligible ID without duplicates
v1_id1 = DropDuplicates(v1['ParticipantID'])

lable_v1_mr, lable_v1_kl = [], []


# rearrange labels, outputs = labels for bml and KL in order
for id in v1_id1:
    q = v1.loc[(v1['ParticipantID'] == id)]
    lable_v1_kl, lable_v1_mr = some_function(q, id, bml, kl, lable_v1_kl, lable_v1_mr, mri_sequence)

# for id in v2_id2:
#     q = v2.loc[(v2['ParticipantID'] == id)]
#
#     lable_v2_kl, lable_v2_mr = some_function(q, id, bml, kl, lable_v2_kl, lable_v2_mr, mri_sequence)
#
# eligible_out1_v2 = pd.concat(lable_v2_kl)
# eligible_out2_v2 = pd.concat(lable_v2_mr)
# eligible_out1_v2 = eligible_out1_v2.drop(columns='Unnamed: 0')
# eligible_out2_v2 = eligible_out2_v2.drop(columns='Unnamed: 0')
# eligible_out1_v2 = eligible_out1_v2.reset_index(drop=True)
# eligible_out2_v2 = eligible_out2_v2.reset_index(drop=True)
#
# temp = eligible_out1_v2[['ID']]
# temp = temp.drop_duplicates()
# temp = temp.reset_index(drop=True)
#
# temp1 = eligible_out2_v2[['ID']]
# temp1 = temp1.drop_duplicates()
# temp1 = temp1.reset_index(drop=True)

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
temp = eligible_out_bml[['ID'] + ['SIDE']]
df = eligible_out_bml.drop(columns=['ID', 'SIDE']).copy()
df = df[df != 1]
df = pd.concat([temp, df], axis=1)
no_bml = df.dropna(how='any', axis=0)
no_bml = no_bml.reset_index(drop=True)

# dividing right and left knee from subjects with no BMLs
Eligible_right_no_bml, Eligible_left_no_bml = division_a(no_bml, 'SIDE', 1, 2)

# finding corresponding KL grade (same id which have no BMLs)
kl_no_bml = []
for i in range(len(no_bml['ID'])):
    temp = eligible_out_kl.loc[(eligible_out_kl['ID'] == no_bml['ID'][i]) &
                               (eligible_out_kl['SIDE'] == no_bml['SIDE'][i])]
    kl_no_bml.append(temp)

# converting to Dataframe format
kl_no_bml = pd.concat(kl_no_bml)

# dividing to 2 part: healthy with no BMLs and OA with no BMLs
hl_nobml, oa_nobml = division_a(kl_no_bml, 'V00XRKL', 0, 1)
hl_nobml_right, hl_nobml_left = division_a(hl_nobml, 'SIDE', 1, 2)
oa_nobml_right, oa_nobml_left = division_a(oa_nobml, 'SIDE', 1, 2)


# Merging all clinical info into one Dataframe
info = info[['ID'] + ['P01BMI'] + ['V00AGE']]
info_sex = info_sex['P02SEX']
clinical_info = pd.concat([info, info_sex], axis=1)

# Finding corresponding clinical info for any subgroup
info_hl = Corresponding(hl_nobml, clinical_info)

