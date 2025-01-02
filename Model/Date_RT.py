import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties

# 数据集文件导入
file_path1 = 'D:/python machine learning/RandomTree_scikit/Date/Data_6.25.xlsx'
file_path2 = 'D:/python machine learning/RandomTree_scikit/Date/Date_4.23_vm.xlsx'  # 改
df = pd.read_excel(file_path1)
df1 = pd.read_excel(file_path2)
print(df.columns.to_list())
print(df.shape)
print(df1.columns.to_list())
print(df1.shape)

column_to_encode1 = 'Agent'
df1_encoded1 = pd.get_dummies(df1[column_to_encode1], prefix=column_to_encode1)
df1 = pd.concat([df1, df1_encoded1], axis=1)
df1.drop(columns=[column_to_encode1], inplace=True)
df1.to_csv('output_data_vm_RT_4.23_test.csv', index=False)  # 改
data1 = pd.read_csv('D:/python machine learning/RandomTree_scikit/Date/output_data_vm_RT_4.23_test.csv')  # 改


# 独热编码处理并分别保存为csv文件
column_to_encode = 'Agent'
df_encoded = pd.get_dummies(df[column_to_encode], prefix=column_to_encode)
df = pd.concat([df, df_encoded], axis=1)
df.drop(columns=[column_to_encode], inplace=True)
df.to_csv('output_data_RT_4.23_test.csv', index=False)  # 改
data = pd.read_csv('D:/python machine learning/RandomTree_scikit/Date/output_data_RT_4.23_test.csv')

# region 相关系数矩阵热力图
# 计算相关系数矩阵
column_labels = {
    'C (%)': 'C',
    'H (%)': 'H',
    'O (%)': 'O',
    'N (%)': 'N',
    'S (%)': 'S',
    'VM (%)': 'VM',
    'Ash (%)': 'Ash',
    'FC (%)': 'FC',
    'Agent_sample': 'A/S',
    'Heating_rate': 'HR',
    'Activation_temp': 'Temp',
    'Activation_time': 'Time',
    'SBET': 'SSA',
    'TPV': 'TPV',
    'Microporous_Ratio': 'MR',
    'Agent_H3PO4': '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
    'Agent_K2CO3': '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$',
    'Agent_K3PO4': '$K_{\mathrm{3}}$P$O_{\mathrm{4}}$',
    'Agent_KOH': 'KOH',
    'Agent_Na2CO3': '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$',
    'Agent_NaOH': 'NaOH',
    'Agent_ZnCl2': 'Zn$Cl_{\mathrm{2}}$',
}

corr_matrix = data.corr()

# 绘制相关系数矩阵热力图
plt.figure(figsize=(6.93, 6.93), dpi=600)
sns.set(font_scale=1.2)

# 自定义调色板
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# 创建 FontProperties 对象，设置字体属性
font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

font_properties = FontProperties(fname=font_path, size=6)
custom_font = fm.FontProperties(fname=font_path)
annot_font = {'fontsize': 5, 'fontweight': 'bold'}

# 绘制热力图
ax = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
            square=True, cbar_kws={"shrink": 0.75}, vmin=-1, vmax=1, linewidths=0.,
            annot_kws=annot_font, xticklabels=[column_labels.get(c, c) for c in corr_matrix.columns],
            yticklabels=[column_labels.get(c, c) for c in corr_matrix.index])

ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_properties, weight='bold', rotation=45, ha='center')
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_properties, weight='bold', rotation=0)

# 获取颜色条对象并设置字体属性
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=6, width=0.5, length=2)  # 修改 labelsize 以更改颜色条刻度标签的字体大小

plt.tight_layout()
plt.show()
# endregion

# 将数据拆分为带有缺失值和不带缺失值的两部分
data_with_missing = data[data.isnull().any(axis=1)]  # 含有缺失值的数据
data_no_missing = data.dropna()                      # 不含缺失值的数据

# region 训练集和测试集划分
# 将数据集分割为训练集和验证集
# n_splits 表示将数据集拆分为几个训练/测试对 train_size=0.8表示训练集所占的整个数据集比例 rando_state随机种子
s = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=80)
# 以活化方式为分层标准合并独热编码
strata_variable = data_no_missing[['Agent_H3PO4', 'Agent_KOH', 'Agent_ZnCl2']]

# 重置索引确保一致性
data_no_missing.reset_index(drop=True, inplace=True)
strata_variable.reset_index(drop=True, inplace=True)

for train_index, test_index in s.split(data_no_missing, strata_variable):
    # print(train_index,test_index)
    strat_train_set = data_no_missing.loc[train_index]
    strat_test_set = data_no_missing.loc[test_index]
# endregion


# 提取特征和目标变量
feature_train = strat_train_set.drop(columns=['SBET', 'TPV', 'Microporous_Ratio'])  # 特征  # 改
label1_train = strat_train_set['SBET']
label2_train = strat_train_set['TPV']
label3_train = strat_train_set['Microporous_Ratio']

feature_test = strat_test_set.drop(columns=['SBET', 'TPV', 'Microporous_Ratio'])  # 特征  # 改
label1_test = strat_test_set['SBET']
label2_test = strat_test_set['TPV']
label3_test = strat_test_set['Microporous_Ratio']

# 目标变量
feature_test_missing = data1.drop(columns=['SBET', 'TPV', 'Microporous_Ratio'])  # 待填充的特征 改

'''
RT_model1 = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_split=2, min_samples_leaf=2,
                                  max_features=2, random_state=48)  # SET
'''
RT_model1 = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, min_samples_leaf=1,
                                  max_features=2, random_state=64)
RT_model2 = RandomForestRegressor(n_estimators=100, random_state=46)
RT_model3 = RandomForestRegressor(n_estimators=80, max_depth=10, min_samples_split=2, min_samples_leaf=2,
                                  max_features=2, random_state=50)  # MIC



'''
# 提取特征和目标变量
feature_train = strat_train_set.drop(columns=['VM', 'Ash', 'FC'])  # 特征  # 改
label1_train = strat_train_set['VM']
label2_train = strat_train_set['Ash']
label3_train = strat_train_set['FC']

feature_test = strat_test_set.drop(columns=['VM', 'Ash', 'FC'])  # 特征  # 改
label1_test = strat_test_set['VM']
label2_test = strat_test_set['Ash']
label3_test = strat_test_set['FC']

feature_test_missing = data1.drop(columns=['VM', 'Ash', 'FC'])  # 待填充的特征

RT_model1 = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=3, min_samples_leaf=1,
                                  max_features=2, random_state=65)   # VM
RT_model2 = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=4, min_samples_leaf=2,
                                  max_features=2, random_state=62)  # Ash
RT_model3 = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_split=3, min_samples_leaf=1,
                                  max_features=3, random_state=66)  # fc
'''

# region 训练集评价指标封装
class Evaluate:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels

# 均方根误差计算
    def rmse(self):
        from sklearn.metrics import mean_squared_error
        prediction = self.model.predict(self.features)
        mse = mean_squared_error(self.labels, prediction)
        rmse = np.sqrt(mse)
        print("-- RMSE --")
        print(f"Root Mean Square Error is: {rmse}")

# 交叉验证并打印均值和标准差
    def cross_validation(self):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.features, self.labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        print("-- Cross Validation --")
        print("Mean: ", rmse_scores.mean())
        print("Standard deviation: ", rmse_scores.std())

# R2分析
    def r2_score(self):
        from sklearn.metrics import r2_score
        label_prediction = self.model.predict(self.features)
        r2 = r2_score(self.labels, label_prediction)
        print("-- R2 Score --")
        print("R2: ", r2)

    def NRMSE(self):
        from sklearn.metrics import mean_squared_error
        mean_feature = np.mean(self.labels)
        prediction = self.model.predict(self.features)
        mse = mean_squared_error(self.labels, prediction)
        rmse = np.sqrt(mse)
        nrmse = rmse/mean_feature
        print("-- NRMSE --")
        print("NRMSE: ", nrmse)
        print("Mean: ", mean_feature)

# 准确率(回归问题不需要)
    '''
    def accuracy(self):
        from sklearn.metrics import accuracy_score
        label_predict = self.model.predict(self.features)
        accuracy = accuracy_score(self.labels, label_predict)
        print("-- Accuracy --")
        print("accuracy: ", accuracy)

    print(f"Trained examples count: {len(features_prepared_train)}")
    '''
# endregion


RT_model1.fit(feature_train, label1_train)


RT_model2.fit(feature_train, label2_train)


RT_model3.fit(feature_train, label3_train)

# region 预测效果
print("\n\n--- VM ----")

print("\n\n-- TRAIN EVALUATION ---")
e5 = Evaluate(RT_model1, feature_train, label1_train)
e5.rmse()
e5.cross_validation()
e5.r2_score()
e5.NRMSE()

# 验证集评估
print("\n\n--- TEST EVALUATION ---")
e6 = Evaluate(RT_model1, feature_test, label1_test)
e6.rmse()
e6.r2_score()
e6.NRMSE()

print("\n\n--- Ash ----")

print("\n\n-- TRAIN EVALUATION ---")
e5 = Evaluate(RT_model2, feature_train, label2_train)
e5.rmse()
e5.cross_validation()
e5.r2_score()
e5.NRMSE()

# 验证集评估
print("\n\n--- TEST EVALUATION ---")
e6 = Evaluate(RT_model2, feature_test, label2_test)
e6.rmse()
e6.r2_score()
e6.NRMSE()

print("\n\n--- FC ----")

print("\n\n-- TRAIN EVALUATION ---")
e5 = Evaluate(RT_model3, feature_train, label3_train)
e5.rmse()
e5.cross_validation()
e5.r2_score()
e5.NRMSE()

# 验证集评估
print("\n\n--- TEST EVALUATION ---")
e6 = Evaluate(RT_model3, feature_test, label3_test)
e6.rmse()
e6.r2_score()
e6.NRMSE()
# endregion

predictions_model1 = RT_model1.predict(feature_train)
predictions_model2 = RT_model2.predict(feature_train)
predictions_model3 = RT_model3.predict(feature_train)

predictions_test_model1 = RT_model1.predict(feature_test)
predictions_test_model2 = RT_model2.predict(feature_test)
predictions_test_model3 = RT_model3.predict(feature_test)

data3 = {
    'Predicted_Train1': predictions_model1,
    'Actual_Train1': label1_train,
    'Predicted_Train2': predictions_model2,
    'Actual_Train2': label2_train,
    'Predicted_Train3': predictions_model3,
    'Actual_Train3': label3_train,
}

'''
'Predicted_Train1': predictions_model1,
'Actual_Train1': label1_train,

'Predicted_Test1': predictions_test_model1,
'Actual_Test1': label1_test,

'Predicted_Test2': predictions_test_model2,
'Actual_Test2': label2_test

'Predicted_Test3': predictions_test_model3,
'Actual_Test3': label3_test,
'''
df = pd.DataFrame(data3)


# 将DataFrame导出为CSV文件
df.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/RF-imputation.csv', index=False)


'''
prediction1 = RT_model1.predict(feature_test)
prediction2 = RT_model2.predict(feature_test)
'''

'''
# prediction3 = RT_model3.predict(feature_test_missing)
prediction1 = RT_model1.predict(feature_test_missing)
prediction2 = RT_model2.predict(feature_test_missing)
prediction3 = RT_model2.predict(feature_test_missing)

df1['VM'] = prediction1
df1['Ash'] = prediction2
df1['FC'] = prediction3

df1.to_csv('D:/python machine learning/RandomTree_scikit/Date/output_data_vm_RT_4.23_test.csv', index=False)  # 改
'''
