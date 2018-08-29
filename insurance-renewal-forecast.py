import tensorflow as tf
import pandas as pd
import numpy as np

print("load policy...")
df_train_set = pd.read_csv('training-set.csv')
df_test_set = pd.read_csv('testing-set.csv')
df_claim = pd.read_csv('claim_0702.csv')
df_policy = pd.read_csv('policy_0702.csv')[[
    'Policy_Number', 'Cancellation', 'Manafactured_Year_and_Month', 'Engine_Displacement_(Cubic_Centimeter)',
    'Imported_or_Domestic_Car', 'qpt', 'Main_Insurance_Coverage_Group', 'Insured_Amount1', 'Insured_Amount2',
    'Insured_Amount3', 'Coverage_Deductible_if_applied', 'Premium', 'Replacement_cost_of_insured_vehicle',
    'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'lia_class', 'plia_acc', 'pdmg_acc', 'fassured', 'ibirth', 'fsex',
    'fmarriage', 'dbirth']]

print("load policy...DONE")

df_policy['Cancellation'] = df_policy['Cancellation'].apply({'Y': 1, ' ': 0}.get)
df_policy['Manafactured_Year_and_Month'] = df_policy['Manafactured_Year_and_Month'].apply(lambda x: 2018 - int(x))

# df_policy['Insured_Amount1'] = df_policy['Insured_Amount1'].apply(lambda x: x / 1000000)
# df_policy['Insured_Amount2'] = df_policy['Insured_Amount2'].apply(lambda x: x / 1000000)
# df_policy['Insured_Amount3'] = df_policy['Insured_Amount3'].apply(lambda x: x / 1000000)
# df_policy['Coverage_Deductible_if_applied'] = df_policy['Coverage_Deductible_if_applied'].apply(lambda x: x / 10000)
# df_policy['Replacement_cost_of_insured_vehicle'] = df_policy['Replacement_cost_of_insured_vehicle'].apply(
#     lambda x: x / 1000)
# df_policy['Multiple_Products_with_TmNewa_(Yes_or_No?)'] = df_policy['Multiple_Products_with_TmNewa_(Yes_or_No?)'].apply(
#     lambda x: x / 1000)
df_policy['Main_Insurance_Coverage_Group'] = df_policy['Main_Insurance_Coverage_Group'].apply(
    {'車責': 0, '竊盜': 1, '車損': 2}.get)
df_policy['ibirth'] = df_policy['ibirth'].str.extract('(19..)', expand=True).fillna(2018)
df_policy['ibirth'] = df_policy['ibirth'].apply(lambda x: 2018 - int(x))
df_policy['fsex'] = df_policy['fsex'].apply({'1': 1, '2': 2, ' ': -1}.get).fillna(0)
df_policy['fmarriage'] = df_policy['fmarriage'].apply({'1': 1, '2': 2, ' ': 0}.get).fillna(0)
df_policy['dbirth'] = df_policy['dbirth'].str.extract('(19..)', expand=True).fillna(2018)
df_policy['dbirth'] = df_policy['dbirth'].apply(lambda x: 2018 - int(x))

print('save Premium_sum.csv')
df_policy[['Policy_Number', 'Premium']].groupby('Policy_Number').sum().to_csv('Premium_sum.csv')
df_Premium_sum = pd.read_csv('Premium_sum.csv')
df_policy.groupby('Policy_Number').mean().rename(columns={'Premium': 'Premium_mean'}).to_csv('policy_mean.csv')
df_policy_mean = pd.read_csv('policy_mean.csv')


print("MERGE...")

df_Policy_claims = pd.merge(df_policy_mean[['Policy_Number', 'Manafactured_Year_and_Month', 'ibirth']],
                            df_claim.groupby('Policy_Number').size().reset_index(), 'left').fillna(0)

df_train_Premium_sum = pd.merge(df_train_set, df_Premium_sum)
df_test_Premium_sum = pd.merge(df_test_set, df_Premium_sum)

df_train = pd.merge(df_train_Premium_sum, df_Policy_claims)
df_test = pd.merge(df_test_Premium_sum, df_Policy_claims)

sample_train = df_train.sample(frac=0.8)
train_x = sample_train.drop(['Policy_Number', 'Next_Premium'], 1)
train_y = sample_train['Next_Premium']

sample_valid = df_train.sample(frac=0.2)
valid_x = sample_valid.drop(['Policy_Number', 'Next_Premium'], 1)
valid_y = sample_valid['Next_Premium']

tf_x = tf.placeholder(tf.float32, [None, 4])
tf_y = tf.placeholder(tf.float32, [None, 1])

tf_layer1 = tf.layers.dense(tf_x, 100, tf.nn.relu)
tf_layer2 = tf.layers.dense(tf_layer1, 10, tf.nn.relu)
tf_output = tf.layers.dense(tf_layer2, 1, tf.nn.relu)

tf_loss = tf.losses.absolute_difference(tf_y, tf_output)
tf_optimizer = tf.train.AdamOptimizer(0.05).minimize(tf_loss)

print("RUN session...")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 1297

for step in range(100):
    for i in range(0, len(sample_train), batch_size):
        output, loss, _ = sess.run([tf_output, tf_loss, tf_optimizer],
                                   {tf_x: train_x[i:i + batch_size], tf_y: train_y[i:i + batch_size, np.newaxis]})
    print(sess.run([tf_output, tf_loss], {tf_x: valid_x, tf_y: valid_y[:, np.newaxis]}))

df_test_set['Next_Premium'] = sess.run(tf_output, {tf_x: df_test.drop(['Policy_Number', 'Next_Premium'], 1)})
df_test_set.to_csv('submission_test.csv', columns=['Policy_Number', 'Next_Premium'], index=False)
print(df_test_set.min())
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML

# load data

df_policy = pd.read_csv("data/policy_0702.csv")
df_train = pd.read_csv("data/training-set.csv")
df_test = pd.read_csv("data/testing-set.csv")
df_test['Next_Premium'] = 0

# Feature Engineering

df_policy['Cancellation_1'] = df_policy['Cancellation'].apply({'Y': 1, ' ': 0}.get)
df_policy['Main_Insurance_Coverage_Group_1'] = df_policy['Main_Insurance_Coverage_Group'].apply(
    {'車責': 0, '竊盜': 1, '車損': 2}.get)

df_policy['ibirth_1'] = df_policy['ibirth'].str.extract('(19..)', expand=True)
df_policy['ibirth_1'].fillna(value=1968, inplace=True)
df_policy['ibirth_1'] = df_policy['ibirth_1'].apply(lambda x: 2017 - int(x))

# total claim for each policy

df_policy['Claim_Counts'] = df_policy['Policy_Number'].map(df_policy['Policy_Number'].value_counts())
df_policy_1 = pd.DataFrame()
df_policy_1 = df_policy[['Policy_Number', 'Claim_Counts', 'Cancellation_1', 'Engine_Displacement_(Cubic_Centimeter)',
                         'Imported_or_Domestic_Car', 'qpt', 'Main_Insurance_Coverage_Group_1',
                         'Insured_Amount1', 'Insured_Amount2', 'Insured_Amount3', 'Coverage_Deductible_if_applied',
                         'Premium', 'Replacement_cost_of_insured_vehicle', 'lia_class', 'plia_acc', 'pdmg_acc',
                         'fassured', 'ibirth_1']]

df_train_1 = pd.merge(df_train, df_policy_1, on="Policy_Number", how="inner")
df_test_1 = pd.merge(df_test, df_policy_1, on="Policy_Number", how="inner")

# Modeling

# import lightgbm as lgb

y = df_train_1['Next_Premium']
features = [f for f in df_train_1.columns if f not in ['Next_Premium', 'Policy_Number']]
data = df_train_1[features]
test = df_test_1[features]

# Split train and validation set

train_x, valid_x, train_y, valid_y = train_test_split(data, y, test_size=0.2, shuffle=True)

# Build LightGBM Model

train_data = lgb.Dataset(train_x, label=train_y)
valid_data = lgb.Dataset(valid_x, label=valid_y)
param_ = {
    'boosting_type': 'gbdt',
    'class_weight': None,
    'colsample_bytree': 0.733333,
    'learning_rate': 0.00764107,
    'max_depth': -1,
    'min_child_samples': 460,
    'min_child_weight': 0.001,
    'min_split_gain': 0.0,
    'n_estimators': 2673,
    'n_jobs': -1,
    'num_leaves': 77,
    'objective': None,
    'random_state': 42,
    'reg_alpha': 0.877551,
    'reg_lambda': 0.204082,
    'silent': True,
    'subsample': 0.949495,
    'subsample_for_bin': 240000,
    'subsample_freq': 1,
    'metric': 'l1'  # aliase for mae
}

# Train model on selected parameters and number of iterations

lgbm = lgb.train(param_,
                 train_data,
                 2500,
                 valid_sets=valid_data,
                 early_stopping_rounds=40,
                 verbose_eval=10
                 )

# predict data

predictions_lgbm_prob = lgbm.predict(test)
result = pd.DataFrame()
result['Policy_Number'] = df_test_1.Policy_Number[:-1]
result['Next_Premium'] = predictions_lgbm_prob

# combine next prenium by mean of same the policy

test1 = df_test.copy()
result1 = result.groupby(result['Policy_Number']).mean()
test1 = test1.merge(result1[['Next_Premium']], on=['Policy_Number'])
submit = test1[['Policy_Number', 'Next_Premium_y']]
submit = submit.rename(index=str, columns={"Next_Premium_y": "Next_Premium"})
submit.to_csv("submit1.csv", index=False)


import tensorflow as tf
import pandas as pd
import numpy as np

# policy_0702 = pd.read_csv('policy_0702.csv')[['Policy_Number', 'Premium']]
testing_set = pd.read_csv('testing-set.csv')

# policy_0702.groupby('Policy_Number').sum().to_csv('policy_0702_sum.csv')

policy_0702_sum = pd.read_csv('policy_0702_sum.csv')

# pd.merge(testing_set, policy_0702_sum, sort=False).drop('Next_Premium').to_csv('submission_test.csv')
# pd.merge(testing_set, policy_0702_sum, sort=False).to_csv('submission_test.csv', columns=['Policy_Number', 'Premium'], index=False)
pd.merge(testing_set, policy_0702_sum, sort=False).drop('Next_Premium', 1).rename(
    columns={'Premium': 'Next_Premium'}).to_csv('submission_test.csv', index=False)
# for index, row in testing_set.iterrows():
#     testing_set['Next_Premium'][index] = policy_0702_sum[['Premium']][policy_0702_sum['Policy_Number'] == row['Policy_Number']].values


# testing_set.to_csv('submission_test.csv')


# last_Policy_Number = ''
# count = -1
#
# for index, row in policy_0702.iterrows():
#     print(index)
#     Policy_Number = row['Policy_Number']
#
#     if Policy_Number == last_Policy_Number:
#         # policy_0702.loc[count]['Premium'] = policy_0702.loc[count]['Premium'].add(row)
#         policy_0702['Policy_Number'].groupby('Policy_Number').sum()
#
#     else:
#         count = count + 1
#         policy_0702.loc[count] = row
#
#     last_Policy_Number = Policy_Number
#
# policy_0702.drop(policy_0702.index[count + 1:]).to_csv('policy_0702_add.csv')

'''
