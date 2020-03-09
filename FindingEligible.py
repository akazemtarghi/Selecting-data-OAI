import pandas as pd
def Eligible_V1(subject,k, m, session, sequences):
    eligible_out = []

    for id in subject:
        content_person = session.loc[(session['ParticipantID'] == id)]

        if sequences[0] in content_person.values and sequences[1] in content_person.values and \
                sequences[2] in content_person.values:

            if not k.loc[(k['ID'] == id) & (k['SIDE'] == 1)].empty:

                if not m.loc[(m['ID'] == id) & (m['SIDE'] == 1)].empty:
                    p2 = session.loc[(session['ParticipantID'] == id) & (
                            session['SeriesDescription'] == sequences[0])]
                    p1 = session.loc[(session['ParticipantID'] == id) &
                                     (session['SeriesDescription'] == sequences[2])]
                    eligible_out.append(p1)
                    eligible_out.append(p2)

            if not k.loc[(k['ID'] == id) & (k['SIDE'] == 2)].empty:

                if not m.loc[(m['ID'] == id) & (m['SIDE'] == 2)].empty:
                    p3 = session.loc[(session['ParticipantID'] == id) & (
                            session['SeriesDescription'] == sequences[1])]
                    p1 = session.loc[(session['ParticipantID'] == id) &
                                     (session['SeriesDescription'] == sequences[2])]
                    eligible_out.append(p1)
                    eligible_out.append(p3)

    eligible_out = pd.concat(eligible_out)
    eligible_out = eligible_out[['Folder'] + ['ParticipantID'] + ['SeriesDescription']]

    eligible_out.drop_duplicates(inplace=True)
    eligible_out = eligible_out.reset_index(drop=True)

    return eligible_out


def Eligible_V2(subject, k, m, session, sequences):
    eligible_out = []

    for id in subject:
        content_person = session.loc[(session['ParticipantID'] == id)]

        if sequences[0] in content_person.values and sequences[1] in content_person.values and \
                sequences[2] in content_person.values:

            if not k.loc[(k['ID'] == str(id)) & (k['SIDE'] == 1)].empty:
                if not m.loc[(m['ID'] == str(id)) & (m['SIDE'] == 1)].empty:
                    if not k.loc[(k['ID'] == str(id)) & (k['SIDE'] == 2)].empty:
                        if not m.loc[(m['ID'] == str(id)) & (m['SIDE'] == 2)].empty:
                            p1 = session.loc[(session['ParticipantID'] == id) &
                                             (session['SeriesDescription'] == sequences[2])]
                            p2 = session.loc[(session['ParticipantID'] == id) & (
                                    session['SeriesDescription'] == sequences[0])]
                            p3 = session.loc[(session['ParticipantID'] == id) & (
                                    session['SeriesDescription'] == sequences[1])]

                            eligible_out.append(p1)
                            eligible_out.append(p2)
                            eligible_out.append(p3)

    eligible_out = pd.concat(eligible_out)
    eligible_out = eligible_out[['Folder'] + ['ParticipantID'] + ['SeriesDescription']]

    eligible_out.drop_duplicates(inplace=True)
    eligible_out = eligible_out.reset_index(drop=True)

    return eligible_out
