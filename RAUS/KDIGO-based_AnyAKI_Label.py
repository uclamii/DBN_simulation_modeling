################################################################################
############ Detecting anyAKI every 24 hours for 7 day horizon #################
################### KDIGO Criteria Based Approach ##############################
################################################################################

# at 1 day
scr_drop_by_1point5_or_more_1days = []
for one,two in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_0):
    if two > one * 1.5 or two >= one + 0.3:
        scr_drop_by_1point5_or_more_1days.append(1)
    else:
        scr_drop_by_1point5_or_more_1days.append(0)

# at 2 days
scr_drop_by_1point5_or_more_2days = []
for one,two in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_1):
    if two > one * 1.5 or two >= one + 0.3:
        scr_drop_by_1point5_or_more_2days.append(1)
    else:
        scr_drop_by_1point5_or_more_2days.append(0)

# at 3 days
scr_drop_by_1point5_or_more_3days = []
for one,two,three,four in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_2,df.scr_24hourperiod_0,df.scr_24hourperiod_1):
    if two > one * 1.5 or two >= three + 0.3 or two >= four + 0.3:
        scr_drop_by_1point5_or_more_3days.append(1)
    else:
        scr_drop_by_1point5_or_more_3days.append(0)

# at 4 days
scr_drop_by_1point5_or_more_4days = []
for one,two,three,four in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_3,df.scr_24hourperiod_1,df.scr_24hourperiod_2):
    if two > one * 1.5 or two >= three + 0.3 or two >= four + 0.3:
        scr_drop_by_1point5_or_more_4days.append(1)
    else:
        scr_drop_by_1point5_or_more_4days.append(0)

# at 5 days
scr_drop_by_1point5_or_more_5days = []
for one,two,three,four in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_4,df.scr_24hourperiod_2,df.scr_24hourperiod_3):
    if two > one * 1.5 or two >= three + 0.3 or two >= four + 0.3:
        scr_drop_by_1point5_or_more_5days.append(1)
    else:
        scr_drop_by_1point5_or_more_5days.append(0)

# at 6 days
scr_drop_by_1point5_or_more_6days = []
for one,two,three,four in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_5,df.scr_24hourperiod_3,df.scr_24hourperiod_4):
    if two > one * 1.5 or two >= three + 0.3 or two >= four + 0.3:
        scr_drop_by_1point5_or_more_6days.append(1)
    else:
        scr_drop_by_1point5_or_more_6days.append(0)

# at 7 days
scr_drop_by_1point5_or_more_7days = []
for one,two,three,four in zip(df.scr_base_enc_valuenum,df.scr_24hourperiod_6,df.scr_24hourperiod_4,df.scr_24hourperiod_5):
    if two > one * 1.5 or two >= three + 0.3 or two >= four + 0.3:
        scr_drop_by_1point5_or_more_7days.append(1)
    else:
        scr_drop_by_1point5_or_more_7days.append(0)

#### Add variables to the dataframe ####

df['aki_progression_1days'] = scr_drop_by_1point5_or_more_1days
df['aki_progression_2days'] = scr_drop_by_1point5_or_more_2days
df['aki_progression_3days'] = scr_drop_by_1point5_or_more_3days
df['aki_progression_4days'] = scr_drop_by_1point5_or_more_4days
df['aki_progression_5days'] = scr_drop_by_1point5_or_more_5days
df['aki_progression_6days'] = scr_drop_by_1point5_or_more_6days
df['aki_progression_7days'] = scr_drop_by_1point5_or_more_7days
