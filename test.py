import tensorflow as tf
import pandas as pd
import numpy as np

df_train_set = pd.read_csv('training-set.csv')
df_test_set = pd.read_csv('testing-set.csv')
df_claim = pd.read_csv('claim_0702(light).csv')
# df_claim = pd.read_csv('data/claim_0702.csv')[[
#     'Nature_of_the_claim', 'Policy_Number', "Driver's_Gender", "Driver's_Relationship_with_Insured", 'DOB_of_Driver',
#     'Marital_Status_of_Driver', 'Accident_Date', 'Paid_Loss_Amount', 'paid_Expenses_Amount', 'Salvage_or_Subrogation?',
#     'At_Fault?', 'Claim_Status_(close,_open,_reopen_etc)', 'Deductible', 'number_of_claimants']]
df_policy = pd.read_csv('policy_0702.csv')[[
    'Policy_Number', 'Cancellation', 'Manafactured_Year_and_Month', 'Engine_Displacement_(Cubic_Centimeter)',
    'Imported_or_Domestic_Car', 'qpt', 'Main_Insurance_Coverage_Group', 'Insured_Amount1', 'Insured_Amount2',
    'Insured_Amount3', 'Coverage_Deductible_if_applied', 'Premium', 'Replacement_cost_of_insured_vehicle',
    'Multiple_Products_with_TmNewa_(Yes_or_No?)', 'lia_class', 'plia_acc', 'pdmg_acc', 'fassured', 'ibirth', 'fsex',
    'fmarriage', 'dbirth']]


# df_policy[['Policy_Number', 'Premium']].groupby('Policy_Number').sum().rename(columns={'Premium': 'Premium_sum'}).to_csv('Premium_sum.csv')
df_Premium_sum = pd.read_csv('Premium_sum.csv')

# df_policy['Cancellation'] = df_policy['Cancellation'].apply({'Y': 1, ' ': 0}.get)
#
# df_policy['Manafactured_Year_and_Month'] = df_policy['Manafactured_Year_and_Month'].apply(lambda x: 2018 - int(x))
#
# df_policy['Engine_Displacement_(Cubic_Centimeter)'] = df_policy['Engine_Displacement_(Cubic_Centimeter)'].apply(
#     lambda x: x / 1000)
#
# df_policy['Insured_Amount1'] = df_policy['Insured_Amount1'].apply(lambda x: x / 1000000)
# df_policy['Insured_Amount2'] = df_policy['Insured_Amount2'].apply(lambda x: x / 1000000)
# df_policy['Insured_Amount3'] = df_policy['Insured_Amount3'].apply(lambda x: x / 1000000)
#
# df_policy['Coverage_Deductible_if_applied'] = df_policy['Coverage_Deductible_if_applied'].apply(lambda x: x / 10000)
#
# df_policy['Premium'] = df_policy['Premium'].apply(lambda x: x / 100000)
#
# df_policy['Replacement_cost_of_insured_vehicle'] = df_policy['Replacement_cost_of_insured_vehicle'].apply(
#     lambda x: x / 1000)
#
# df_policy['Multiple_Products_with_TmNewa_(Yes_or_No?)'] = df_policy['Multiple_Products_with_TmNewa_(Yes_or_No?)'].apply(
#     lambda x: x / 1000)
#
# df_policy['Main_Insurance_Coverage_Group'] = df_policy['Main_Insurance_Coverage_Group'].apply(
#     {'車責': 0, '竊盜': 1, '車損': 2}.get)
#
# df_policy['ibirth'] = df_policy['ibirth'].str.extract('(19..)', expand=True).fillna(2018)
# df_policy['ibirth'] = df_policy['ibirth'].apply(lambda x: 2018 - int(x))
#
# df_policy['fsex'] = df_policy['fsex'].apply({'1': 1, '2': 2, ' ': -1}.get).fillna(0)
# df_policy['fmarriage'] = df_policy['fmarriage'].apply({'1': 1, '2': 2, ' ': 0}.get).fillna(0)
#
# df_policy['dbirth'] = df_policy['dbirth'].str.extract('(19..)', expand=True).fillna(2018)
# df_policy['dbirth'] = df_policy['dbirth'].apply(lambda x: 2018 - int(x))

# df_claim['Paid_Loss_Amount'] = df_claim['Paid_Loss_Amount'].apply(lambda x: x / 100000)
# 
# df_claim['paid_Expenses_Amount'] = df_claim['paid_Expenses_Amount'].apply(lambda x: x / 10000)
# 
# df_claim['Salvage_or_Subrogation?'] = df_claim['Salvage_or_Subrogation?'].apply(lambda x: x / 100000)
# 
# df_claim['Deductible'] = df_claim['Deductible'].apply(lambda x: x / 10000)

# df_policy[['Policy_Number', 'Premium']].groupby('Policy_Number').sum().rename(columns={'Premium': 'Next_Premium'}).to_csv('Premium_sum.csv')
# df_Premium_sum = pd.read_csv('Premium_sum.csv')
# df_policy.groupby('Policy_Number').mean().to_csv('policy_mean.csv')
df_policy_mean = pd.read_csv('policy_mean.csv')
# df_claim.groupby('Policy_Number').mean().to_csv('claim_mean.csv')
# df_claim_mean = pd.read_csv('claim_mean.csv')
# print(df_claim_mean.max())

# df_policy_mean_Premium_sum = pd.merge(df_policy_mean, df_Premium_sum)
# df_policy_claim_mean_Premium_sum = pd.merge(pd.merge(df_policy_mean, df_claim_mean), df_Premium_sum)
# df_policy_mean_Premium_sum['Next_Premium_difference'] = df_policy_mean_Premium_sum['Next_Premium'] - df_policy_mean_Premium_sum['Premium']


# df_claim_mean['claims'] = df_claim.groupby('Policy_Number').size()
# df_claim['claims'] = df_claim.groupby('Policy_Number')['Policy_Number'].transform('count')
# df_claim_mean['claims'] = claims['claims']
# print(df_claim.groupby('Policy_Number').size().reset_index())

df_train_Premium_sum = pd.merge(df_train_set, df_Premium_sum)
df_test_Premium_sum = pd.merge(df_test_set, df_Premium_sum)

df_train_Premium_sum['Next_Premium_difference'] = df_train_Premium_sum['Next_Premium'] - df_train_Premium_sum['Premium_sum']
df_Policy_claims = pd.merge(df_policy_mean[['Policy_Number', 'Premium']], df_claim.groupby('Policy_Number').size().reset_index(), 'left').fillna(0)

# df_feature = pd.merge(df_Premium_sum, df_claim.groupby('Policy_Number').size().reset_index(), 'left').fillna(0)
# df_feature = pd.merge(df_policy_mean_Premium_sum, df_claim.groupby('Policy_Number').size().reset_index(), 'left').fillna(0)
df_train = pd.merge(df_train_Premium_sum, df_Policy_claims)
df_test = pd.merge(df_test_Premium_sum, df_Policy_claims)
sample_train = df_train.sample(frac=0.8)
train_x = sample_train.drop(['Policy_Number', 'Next_Premium', 'Next_Premium_difference'], 1)
train_y = sample_train['Next_Premium_difference']
# train_x = sample_train.drop(['Policy_Number', 'Next_Premium'], 1)
# train_y = sample_train['Next_Premium']

sample_valid = df_train.sample(frac=0.2)
valid_x = sample_valid.drop(['Policy_Number', 'Next_Premium', 'Next_Premium_difference'], 1)
valid_y = sample_valid['Next_Premium_difference']
# valid_x = sample_valid.drop(['Policy_Number', 'Next_Premium'], 1)
# valid_y = sample_valid['Next_Premium']

tf_x = tf.placeholder(tf.float32, [None, 3])
tf_y = tf.placeholder(tf.float32, [None, 1])

# layer1 = tf.layers.dense(tf_x, 7, tf.nn.relu)
# layer2 = tf.layers.dense(layer1, 8, tf.nn.relu)
# layer3 = tf.layers.dense(layer2, 2, tf.nn.relu)
# layer4 = tf.layers.dense(layer3, 2, tf.nn.relu)
# output = tf.layers.dense(layer4, 1)
layer1 = tf.layers.dense(tf_x, 125, tf.nn.relu)
layer2 = tf.layers.dense(layer1, 25, tf.nn.relu)
layer3 = tf.layers.dense(layer2, 5, tf.nn.relu)
output = tf.layers.dense(layer3, 1)

tf_loss = tf.losses.absolute_difference(tf_y, output)
optimizer = tf.train.AdamOptimizer(0.1).minimize(tf_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 1297

# print(sess.run(layer1, {tf_x: valid_x, tf_y: valid_y[:, np.newaxis]}))
for step in range(100):

    for i in range(0, len(sample_train), batch_size):
        # print(i)
        prediction, loss, _ = sess.run([output, tf_loss, optimizer],
                                       {tf_x: train_x[i:i + batch_size], tf_y: train_y[i:i + batch_size, np.newaxis]})
        # print(loss)
    print(sess.run([output, tf_loss], {tf_x: valid_x, tf_y: valid_y[:, np.newaxis]}))

# df_test_set['difference'] = sess.run(output, {tf_x: df_test.drop(['Policy_Number', 'Next_Premium'], 1)})
df_test_set['output'] = sess.run(output, {tf_x: df_test.drop(['Policy_Number', 'Next_Premium'], 1)})
print(df_test_set['output'])
df_test_set['Next_Premium'] = (df_train_Premium_sum['Premium_sum'] + df_test_set['output']).abs()
df_test_set.to_csv('submission_test.csv', columns=['Policy_Number', 'Next_Premium'], index=False)
# print(df_test_set['difference'])
