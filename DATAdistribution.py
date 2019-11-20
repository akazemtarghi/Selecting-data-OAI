import matplotlib.pyplot as plt
import pandas as pd
def some_function(q,id,bml,kl,lable_v1_kl,lable_v1_mr,mri_sequence):

    if mri_sequence[0] in q.values and mri_sequence[1] in q.values:
        s1 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 1)]
        s2 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 2)]
        s3 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 1)]
        s4 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 2)]
        lable_v1_kl.append(s1)
        lable_v1_kl.append(s2)
        lable_v1_mr.append(s3)
        lable_v1_mr.append(s4)
        return lable_v1_kl,lable_v1_mr

    if mri_sequence[0] in q.values:
        s1 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 1)]
        s2 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 1)]
        lable_v1_kl.append(s1)
        lable_v1_mr.append(s2)
        return lable_v1_kl,lable_v1_mr

    if mri_sequence[1] in q.values:

        s1 = kl.loc[(kl['ID'] == id) & (kl['SIDE'] == 2)]
        s2 = bml.loc[(bml['ID'] == id) & (bml['SIDE'] == 2)]
        lable_v1_kl.append(s1)
        lable_v1_mr.append(s2)

        return lable_v1_kl,lable_v1_mr





filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Final_output_v1.csv'
filename2 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/Final_output_v2.csv'
v1 = pd.read_csv(filename1)
v2 = pd.read_csv(filename2)
filename1 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/kl.csv'
filename2 = 'C:/Users/Amir Kazemtarghi/Documents/MASTER THESIS/Coding/bml.csv'
kl = pd.read_csv(filename1)
bml = pd.read_csv(filename2)

v1_id1 = v1['ParticipantID'].copy()
v1_id1 = v1_id1.drop_duplicates()
v1_id1 = v1_id1.reset_index(drop=True)
v2_id2 = v2['ParticipantID'].copy()
v2_id2 = v2_id2.drop_duplicates()
v2_id2 = v2_id2.reset_index(drop=True)
a=1
mri_sequence = ['SAG_3D_DESS_RIGHT', 'SAG_3D_DESS_LEFT', 'Bilateral PA Fixed Flexion Knee']
lable_v1_mr = []
lable_v1_kl = []



for id in v1_id1:
    q = v1.loc[(v1['ParticipantID'] == id)]

    lable_v1_kl,lable_v1_mr = some_function(q,id,bml,kl,lable_v1_kl,lable_v1_mr,mri_sequence)



eligible_out1 = pd.concat(lable_v1_kl)
eligible_out2 = pd.concat(lable_v1_mr)

eligible_out1 = eligible_out1.drop(columns='Unnamed: 0')
eligible_out2 = eligible_out2.drop(columns='Unnamed: 0')


df = eligible_out2.drop(columns=['ID','SIDE'])


df_asint = df.astype(int)
coocc = df_asint.T.dot(df_asint)
print(coocc)

zero = (df == 0).sum(axis=0)






