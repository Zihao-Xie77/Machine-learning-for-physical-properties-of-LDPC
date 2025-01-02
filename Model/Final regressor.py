import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
from sklearn.inspection import PartialDependenceDisplay
from matplotlib.ticker import FormatStrFormatter
from sklearn.inspection import permutation_importance
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import KFold

# 加载数据集
date = pd.read_csv('../Date/output_data_6.11_test.csv')

# 将数据集分割为训练集和验证集
# n_splits 表示将数据集拆分为几个训练/测试对 train_size=0.8表示训练集所占的整个数据集比例 rando_state随机种子
s = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=77)
# 以活化方式为分层标准合并独热编码
strata_variable = date[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                        'Agent_ZnCl2']]

for train_index, test_index in s.split(date, strata_variable):
    # print(train_index,test_index)
    strat_train_set = date.loc[train_index]
    strat_test_set = date.loc[test_index]

# 机器学习管道，也可以在date.py中完成并进行可视化分析
my_pipeline = Pipeline([
    # ('imputer', SimpleImputer(strategy="median")),  # 缺失项填补
    ('std_scaler', StandardScaler())    # 特征值放缩
])

# 训练集特征和标签的选取
features_train = strat_train_set.drop(['SBET (cm2/g)', 'TPV (cm3/g)',
                                       'Microporous_Ratio (%)'], axis=1)

label1_train = strat_train_set['SBET (cm2/g)']
label2_train = strat_train_set['TPV (cm3/g)']
label3_train = strat_train_set['Microporous_Ratio (%)']

# 测试集特征和标签的选取
features_test = strat_test_set.drop(['SBET (cm2/g)', 'TPV (cm3/g)',
                                       'Microporous_Ratio (%)'], axis=1)

label1_test = strat_test_set['SBET (cm2/g)']
label2_test = strat_test_set['TPV (cm3/g)']
label3_test = strat_test_set['Microporous_Ratio (%)']

# 通过管道对特征进行预处理
features_prepared_train = my_pipeline.fit_transform(features_train)
features_prepared_test = my_pipeline.fit_transform(features_test)

# 对实验数据采用相同的dataform放缩
scaler = StandardScaler()
scaler.fit(features_train)

data_exp = pd.read_excel(r'D:\python machine learning\RandomTree_scikit\Date\Data_exprriments_8.7.xlsx')

data_exp = pd.DataFrame(data_exp)

data_prepared_exp = scaler.transform(data_exp)



# 模型选择,决策树的数量
'''
model1GBR_416 = GradientBoostingRegressor(n_estimators=160, learning_rate=0.055, max_depth=7, max_features=0.42,
                                          min_samples_split=4, min_samples_leaf=4, random_state=54)
'''
# 随机种子 77 77
model1GBR_416 = GradientBoostingRegressor(n_estimators=170, learning_rate=0.2, max_depth=6, max_features=6,
                                          min_samples_split=5, min_samples_leaf=4, random_state=77)
# 随机种子 77 78
model2GBR_416 = GradientBoostingRegressor(n_estimators=170, learning_rate=0.56, max_depth=7, max_features=7,
                                          min_samples_split=4, min_samples_leaf=6, random_state=78)
# 随机种子 79 77 0.83 记得调数据划分随机种子
'''
model3GBR_416 = GradientBoostingRegressor(n_estimators=160, learning_rate=0.044, max_depth=7, max_features=5,
                                          min_samples_split=4, min_samples_leaf=4, random_state=77)
'''
'''
# 这个不大行
model3GBR_416 = GradientBoostingRegressor(n_estimators=160, learning_rate=0.5, max_depth=7, max_features=5,
                                          min_samples_split=4, min_samples_leaf=4, random_state=77)
'''
# 随机种子 79 77 0.81 记得调数据划分随机种子
model3GBR_416 = GradientBoostingRegressor(n_estimators=160, learning_rate=0.2, max_depth=6, max_features=4,
                                          min_samples_split=5, min_samples_leaf=4, random_state=77)

# 训练预处理好的数据
model1GBR_416.fit(features_prepared_train, label1_train)
model2GBR_416.fit(features_prepared_train, label2_train)
model3GBR_416.fit(features_prepared_train, label3_train)

# 训练数据导出
label_exp1 = model1GBR_416.predict(data_prepared_exp)
label_exp2 = model2GBR_416.predict(data_prepared_exp)

data = {
    "predicted SSA": label_exp1,
    "predicted TPV": label_exp2,
}

df_data = pd.DataFrame(data)

# 将DataFrame导出为CSV文件
df_data.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/experiment_8.7.csv', index=False)

# KOH下的特征重要性排列

# region KOH条件下的特征重要性
# 筛选KOH条件下的数据子集
KOH_subset = date[date['Agent_KOH'] == 1]

# 提取特征和标签
features_KOH = KOH_subset.drop(['SBET (cm2/g)', 'TPV (cm3/g)', 'Microporous_Ratio (%)'], axis=1)
label1_KOH = KOH_subset['SBET (cm2/g)']
label2_KOH = KOH_subset['TPV (cm3/g)']

# 使用之前的管道对特征进行预处理
features_prepared_KOH = my_pipeline.transform(features_KOH)

# 计算KOH条件下特征的重要性（使用排列重要性）
result1 = permutation_importance(model1GBR_416, features_prepared_KOH, label1_KOH, n_repeats=10, random_state=42)
result2 = permutation_importance(model2GBR_416, features_prepared_KOH, label2_KOH, n_repeats=10, random_state=42)

# 创建一个DataFrame来显示特征和它们的重要性
feature_names = features_KOH.columns

feature_importance_df1 = pd.DataFrame({'Feature': feature_names, 'Importance': result1.importances_mean})
feature_importance_df2 = pd.DataFrame({'Feature': feature_names, 'Importance': result2.importances_mean})

# 导入csv文件
feature_importance_df1.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/fi_KOH_SSA.csv', index=False)
feature_importance_df2.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/fi_KOH_TPV.csv', index=False)
# endregion


# region H3PO4条件下的特征重要性
# MP时再打开
# 筛选KOH条件下的数据子集
H3PO4_subset = date[date['Agent_H3PO4'] == 1]

# 提取特征和标签
features_H3PO4 = H3PO4_subset.drop(['SBET (cm2/g)', 'TPV (cm3/g)', 'Microporous_Ratio (%)'], axis=1)
label3_H3PO4 = H3PO4_subset['Microporous_Ratio (%)']

# 使用之前的管道对特征进行预处理
features_prepared_H3PO4 = my_pipeline.transform(features_H3PO4)

# 计算KOH条件下特征的重要性（使用排列重要性）
result3 = permutation_importance(model3GBR_416, features_prepared_H3PO4, label3_H3PO4, n_repeats=10, random_state=42)

# 创建一个DataFrame来显示特征和它们的重要性
feature_names = features_H3PO4.columns

feature_importance_df1 = pd.DataFrame({'Feature': feature_names, 'Importance': result3.importances_mean})

# 导入csv文件
feature_importance_df1.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/fi_H3PO4_MP.csv', index=False)

# endregion


# region 偏依赖图

# 选择要绘制偏依赖图的特征索引
feature_index = 11

fig, ax = plt.subplots(figsize=(8.66, 6.93), dpi=500)
display = PartialDependenceDisplay.from_estimator(model3GBR_416, features_prepared_H3PO4, [feature_index],
                                                  kind='average', ax=ax,
                                                  line_kw={"color": "green"},
                                                  pd_line_kw={"alpha": 1.0})
plt.show()

# 生成偏依赖数据
pdp_results = display.pd_results[0]  # 获取偏依赖结果
pdp_values = pdp_results['grid_values'][0]  # 获取特征值
pdp_avg = pdp_results['average']  # 获取平均偏依赖值

# 确保提取的结果是一维数组
if pdp_avg.ndim > 1:
    pdp_avg = pdp_avg.flatten()

# LOESS 平滑处理
frac = 0  # 平滑程度，可以根据需要调整
smoothed_pdp_avg = lowess(pdp_avg, pdp_values, frac=frac)[:, 1]

# 创建 DataFrame 保存数据
pdp_df = pd.DataFrame({
    'Feature Value': pdp_values,
    'Partial Dependence': smoothed_pdp_avg
})

# 保存为 CSV 文件
pdp_df.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/PD_H3PO4_MP.csv', index=False)
# endregion


# region 超参数调整
# 超参数调整
'''
# 定义参数网络
param_grid = {
    'n_estimators': np.arange(50, 160, 10),
    'learning_rate': np.arange(0.01, 0.1, 0.01),
    'max_depth': np.arange(7, 15, 2),
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': np.arange(1, 5, 1),
    'max_features': ['sqrt', 'log2']
}

# 初始化 GridSearchCV
grid_search = GridSearchCV(estimator=model3GBR_416, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, error_score='raise')

# 在数据上拟合 GridSearchCV
grid_search.fit(features_prepared_train, label3_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)
'''
# endregion

# region RMSE和R2封装
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

    '''
# 打印交叉结果到csv文件内
    def cross_validation1(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=77)
        relative_errors = []
        fold_index = 0

        for train_index, test_index in kf.split(self.features):
            fold_index += 1
            X_train, X_test = features_np[train_index], features_np[test_index]
            y_train, y_test = labels_np[train_index], labels_np[test_index]

            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_test)

            for true, pred in zip(y_test, predictions):
                relative_error = (true - pred) / true if true != 0 else 0
                relative_errors.append(
                    {'Fold': fold_index, 'True Value': true, 'Predicted Value': pred, 'Relative Error': relative_error})

        relative_errors_df = pd.DataFrame(relative_errors)
        relative_errors_df.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/Kflod_RMSE', index=False)
        print("-- Cross Validation --")
        print("Relative errors have been saved to 'Kflod_RMSE'.")
    '''

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

# region 训练集和测试集评估
print("\n\n--- SBET ----")

# 训练集评估
print("\n\n-- TRAIN EVALUATION ---")
e1 = Evaluate(model1GBR_416, features_prepared_train, label1_train)
e1.rmse()
e1.cross_validation()
e1.r2_score()

# 验证集评估
print("\n\n--- TEST EVALUATION ---")
e2 = Evaluate(model1GBR_416, features_prepared_test, label1_test)
e2.rmse()
e2.r2_score()

print("\n\n--- VPT ----")

# 训练集评估
print("\n\n-- TRAIN EVALUATION ---")
e3 = Evaluate(model2GBR_416, features_prepared_train, label2_train)
e3.rmse()
e3.cross_validation()
e3.r2_score()

# 验证集评估
print("\n\n--- TEST EVALUATION ---")
e4 = Evaluate(model2GBR_416, features_prepared_test, label2_test)
e4.rmse()
e4.r2_score()

print("\n\n--- Microporous_Ratio ----")

print("\n\n-- TRAIN EVALUATION ---")
e5 = Evaluate(model3GBR_416, features_prepared_train, label3_train)
e5.rmse()
e5.cross_validation()
e5.r2_score()

# 验证集评估
print("\n\n--- TEST EVALUATION ---")
e6 = Evaluate(model3GBR_416, features_prepared_test, label3_test)
e6.rmse()
e6.r2_score()
# R2分析

'''
from sklearn.metrics import r2_score
label_prediction = model1.predict(features_prepared_test)
r2 = r2_score(label1_test, label_prediction)
print("-- R2 Score --")
print("SBET R2: ", r2)
'''

# 训练好的模型对象保存至文件中
dump(model1GBR_416, 'model1GBR_SSA.1joblib_6.10')
dump(model2GBR_416, 'model2GBR_TPV.1joblib_6.10')
dump(model3GBR_416, 'model2GBR_MP.1joblib_6.10')

print("\n\nModel Saved.")
print("----- Program Finished -----")
# endregion

predictions_model1 = model1GBR_416.predict(features_prepared_train)
predictions_model2 = model2GBR_416.predict(features_prepared_train)
predictions_model3 = model3GBR_416.predict(features_prepared_train)

predictions_test_model1 = model1GBR_416.predict(features_prepared_test)
predictions_test_model2 = model2GBR_416.predict(features_prepared_test)
predictions_test_model3 = model3GBR_416.predict(features_prepared_test)

# 预测拟合散点导出
data = {
    'Predicted_Test1': predictions_model3,
    'Actual_Test1': label3_train,
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
df = pd.DataFrame(data)

# 将DataFrame导出为CSV文件
df.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/PA-GBR.csv', index=False)

corr_matrix = date.corr()

# 相关系数矩阵热力图

# region 绘制相关系数矩阵热力图
plt.figure(figsize=(20, 20))
sns.set(font_scale=1.2)

# 自定义调色板
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# 绘制热力图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
            square=True, cbar_kws={"shrink": 0.75}, vmin=-1, vmax=1, linewidths=0.5)

plt.tight_layout()
plt.show()
# endregion


'''
# region 偏相关图
# 特征索引
feature_index = 0

fig, ax = plt.subplots(figsize=(8.66, 6.93), dpi=500)
display = PartialDependenceDisplay.from_estimator(model2GBR_416, features_prepared_train, [feature_index],
                                        kind='average', ax=ax,
                                        line_kw={"color": "green"},  # 线条颜色设置为黑色
                                        pd_line_kw={"alpha": 1.0},  # 线条透明度设置为不透明
                                        # pdp_iso_kw={"alpha": 0.1},  # 阴影透明度设置为较低
                                        )
plt.xlabel('Feature C', fontproperties='Arial', size=15, weight='bold')
plt.ylabel('Partial Dependence', fontproperties='Arial', size=15, weight='bold')

plt.yticks(fontproperties='Arial', size=15, weight='bold')
plt.xticks(fontproperties='Arial', size=15, weight='bold')
plt.tight_layout()
plt.show()
# endregion
'''


# region SBET训练集和测试集的预测-实际拟合线
# SBET
model = LinearRegression()
model.fit(predictions_model1.reshape(-1, 1), label1_train)
plt.figure(figsize=(8.66, 8.66), dpi=500)

# 绘制散点图
plt.scatter(predictions_model1, label1_train)
plt.scatter(predictions_test_model1, label1_test)

# 绘制拟合线
x_values = np.linspace(min(predictions_model1), max(predictions_model1), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Fitted Line')
plt.plot(x_values, x_values, color='green', linestyle='--')
plt.xlim(600, 2400)
plt.ylim(600, 2400)

plt.xlabel('Predicted SSA', fontproperties='Arial', size=15, weight='bold')
plt.ylabel('Actual SSA', fontproperties='Arial', size=15, weight='bold')

plt.yticks(fontproperties='Arial', size=15, weight='bold')
plt.xticks(fontproperties='Arial', size=15, weight='bold')

# 查找特定字体路径
font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

# 创建 FontProperties 对象，设置字体属性
font_properties = FontProperties(fname=font_path, size=15, weight='bold')

plt.legend(prop=font_properties, loc='upper left')
plt.tight_layout()
plt.show()

# TPV
model = LinearRegression()
model.fit(predictions_model2.reshape(-1, 1), label2_train)
plt.figure(figsize=(8.66, 8.66), dpi=500)

# 绘制散点图
plt.scatter(predictions_model2, label2_train)
plt.scatter(predictions_test_model2, label2_test)

# 绘制拟合线
x_values = np.linspace(min(predictions_model2), max(predictions_model2), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Fitted Line')
plt.plot(x_values, x_values, color='green', linestyle='--')
plt.xlim(0.1, 1.6)
plt.ylim(0.1, 1.6)

plt.xlabel('Predicted TPV', fontproperties='Arial', size=15, weight='bold')
plt.ylabel('Actual TPV', fontproperties='Arial', size=15, weight='bold')

plt.yticks(fontproperties='Arial', size=15, weight='bold')
plt.xticks(fontproperties='Arial', size=15, weight='bold')

# 查找特定字体路径
font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

# 创建 FontProperties 对象，设置字体属性
font_properties = FontProperties(fname=font_path, size=15, weight='bold')

plt.legend(prop=font_properties, loc='upper left')
plt.tight_layout()
plt.show()


# MIC
model = LinearRegression()
model.fit(predictions_model3.reshape(-1, 1), label3_train)
plt.figure(figsize=(8.66, 8.66), dpi=500)

# 绘制散点图
plt.scatter(predictions_model3, label3_train)
plt.scatter(predictions_test_model3, label3_test)

# 绘制拟合线
x_values = np.linspace(min(predictions_model3), max(predictions_model3), 100)
y_values = model.predict(x_values.reshape(-1, 1))
plt.plot(x_values, y_values, color='red', label='Fitted Line')
plt.plot(x_values, x_values, color='green', linestyle='--')
plt.xlim(10, 95)
plt.ylim(10, 95)

plt.xlabel('Predicted MP', fontproperties='Arial', size=15, weight='bold')
plt.ylabel('Actual MP', fontproperties='Arial', size=15, weight='bold')

plt.yticks(fontproperties='Arial', size=15, weight='bold')
plt.xticks(fontproperties='Arial', size=15, weight='bold')

# 查找特定字体路径
font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

# 创建 FontProperties 对象，设置字体属性
font_properties = FontProperties(fname=font_path, size=15, weight='bold')

plt.legend(prop=font_properties, loc='upper left')
plt.tight_layout()
plt.show()
# endregion

# region 特征重要性排列类
class FeatureImportancePlotter:
    def __init__(self, feature_importances, custom_labels):
        """
        初始化特征重要性绘图器

        :param feature_importances: 特征重要性数组
        :param custom_labels: 特征标签列表
        """
        self.feature_importances = feature_importances
        self.custom_labels = custom_labels
        self.groups = {
            'Elemental composition': ['C', 'H', 'O', 'N', 'S'],
            'Proximate composition': ['VM', 'Ash', 'FC'],
            'Pyrolysis conditions': ['HR', 'Temp', 'Time'],
            'Chemical agents': ['A/S', '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$', '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$',
                                'KOH', '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$',
                                'NaOH', 'Zn$Cl_{\mathrm{2}}$']
        }

    def calculate_group_importances(self):
        """
        计算每个分组的特征重要性

        :return: 包含每个分组特征重要性的字典
        """
        group_importances = {}
        for group, features in self.groups.items():
            group_importance = sum(self.feature_importances[self.custom_labels.index(f)] for f in features)
            group_importances[group] = group_importance
        return group_importances

    def plot_group_importances(self):
        """
        绘制分组特征贡献饼状图
        """
        group_importances = self.calculate_group_importances()

        labels = list(group_importances.keys())
        sizes = list(group_importances.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        plt.figure(figsize=(8, 8), dpi=500)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.axis('equal')  # 使饼图为正圆形

        plt.title('Grouped Feature Importance', fontproperties='Arial', fontsize=15, fontweight='bold')
        plt.show()
# endregion

# 活化剂特征重要性导出
# region 特征重要性图
# SBET
# 获取特征重要性评分
feature_importances = (model1GBR_416.feature_importances_)


custom_labels1 = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

# 将数据保存到DataFrame中
dd1 = pd.DataFrame({
    'Feature': custom_labels1,
    'ImportanceSSA': feature_importances
})

# 将DataFrame保存为CSV文件
dd1.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/feature_importances.csv', index=False)

'''
custom_labels = ['C (%)', 'H (%)', 'O (%)', 'N (%)', 'S (%)', 'VM (%)', 'Ash (%)', 'FC (%)',
                 'A/S', 'Hr (℃/min)', 'Temp (℃)', 'Time (min)',
                 'H\u2083PO\u2084', 'K\u2082C\u2082O\u2084', 'K\u2082CO\u2083', 'KOH',
                  'Na\u2082CO\u2083', 'NaOH', 'ZnCl\u2082']
'''

# 定义感兴趣的特征列表
interested_features = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
# 获取感兴趣特征在 custom_labels 中的索引
interested_indices = [custom_labels1.index(feature) for feature in interested_features]
# 提取感兴趣特征的重要性
interested_importances = feature_importances[interested_indices]

custom_labels2 = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

# 对特征重要性进行排序
sorted_indices = np.argsort(interested_importances)
sorted_importances = interested_importances[sorted_indices]
sorted_labels = [custom_labels2[i] for i in sorted_indices]


# 绘制特征贡献图
plt.figure(figsize=(8.66, 6.93), dpi=500)

bars = plt.barh(range(len(interested_importances)), sorted_importances, align='center', color='darkturquoise')
plt.xlabel('Feature Importance of SSA', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=6)  # 例如，生成 10个等间距刻度
plt.xticks(x_ticks, fontsize=15)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置保留两位小数

plt.tight_layout()
plt.show()

plotter = FeatureImportancePlotter(feature_importances, custom_labels1)
plotter.plot_group_importances()

# TPV
# 获取特征重要性评分
feature_importances = (model2GBR_416.feature_importances_)

custom_labels = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

# 将数据保存到DataFrame中
dd2 =pd.read_csv('D:/python machine learning/RandomTree_scikit/outputdate/feature_importances.csv')
dd2['importanceTPV'] = feature_importances

# 将DataFrame保存为CSV文件
dd2.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/feature_importances.csv', index=False)

# 定义感兴趣的特征列表
interested_features = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
# 获取感兴趣特征在 custom_labels 中的索引
interested_indices = [custom_labels.index(feature) for feature in interested_features]
# 提取感兴趣特征的重要性
interested_importances = feature_importances[interested_indices]

custom_labels = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

# 对特征重要性进行排序
sorted_indices = np.argsort(interested_importances)
sorted_importances = interested_importances[sorted_indices]
sorted_labels = [custom_labels[i] for i in sorted_indices]

# 绘制特征贡献图
plt.figure(figsize=(8.66, 6.93), dpi=500)
bars = plt.barh(range(len(interested_importances)), sorted_importances, align='center', color='orange')
plt.xlabel('Feature Importance of TPV', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=6)  # 例如，生成 10个等间距刻度
plt.xticks(x_ticks, fontsize=15)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置保留两位小数

plt.tight_layout()
plt.show()

# MIC
# 获取特征重要性评分
feature_importances = (model3GBR_416.feature_importances_)

custom_labels = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
# 将数据保存到DataFrame中
dd3 =pd.read_csv('D:/python machine learning/RandomTree_scikit/outputdate/feature_importances.csv')
dd3['importanceMP'] = feature_importances

# 将DataFrame保存为CSV文件
dd3.to_csv('D:/python machine learning/RandomTree_scikit/outputdate/feature_importances.csv', index=False)

# 定义感兴趣的特征列表
interested_features = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
# 获取感兴趣特征在 custom_labels 中的索引
interested_indices = [custom_labels.index(feature) for feature in interested_features]
# 提取感兴趣特征的重要性
interested_importances = feature_importances[interested_indices]

custom_labels = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

# 对特征重要性进行排序
sorted_indices = np.argsort(interested_importances)
sorted_importances = interested_importances[sorted_indices]
sorted_labels = [custom_labels[i] for i in sorted_indices]

# 绘制特征贡献图
plt.figure(figsize=(8.66, 6.93), dpi=500)
bars = plt.barh(range(len(interested_importances)), sorted_importances, align='center', color='sandybrown')
plt.xlabel('Feature Importance of MP', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=6)  # 例如，生成 10个等间距刻度
plt.xticks(x_ticks, fontsize=15)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 设置保留两位小数

plt.tight_layout()
plt.show()

'''
custom_labels = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$','$K_{\mathrm{3}}$P$O_{\mathrm{4}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

# 对特征重要性进行排序
sorted_indices = np.argsort(feature_importances)
sorted_importances = feature_importances[sorted_indices]
sorted_labels = [custom_labels[i] for i in sorted_indices]

# 绘制特征贡献图
plt.figure(figsize=(8.66, 6.93), dpi=500)
bars = plt.barh(range(len(feature_importances)), sorted_importances, align='center', color='sandybrown')
plt.xlabel('Feature Importance of MP', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=11)  # 例如，生成 10个等间距刻度
plt.xticks(x_ticks, fontsize=15)
plt.tight_layout()
plt.show()
'''
# endregion


'''
data_label1 = pd.DataFrame(features_train)
data_label2 = pd.DataFrame(features_train)
data_label3 = pd.DataFrame(features_train)

# 选择指定的四列特征
selected_columns = [10, 8, 0, 2]  # 假设你想要选择第 1、3、5 和 7 列特征


# 将预测目标 'SBET (cm2/g)' 添加到选定的四列特征中
data_label1['SBET (cm2/g)'] = label1_train
data_label2['TPV (cm3/g)'] = label2_train
data_label3['Microporous_Ratio (%)'] = label3_train


'''

# 绘制特征与预测目标之间的折线图
'''
# SBET
for feature in data_label1.columns[:-1]:  # 最后一列是预测目标，不需要绘制
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=feature, y=data_label1.columns[-1], data=data_label1)
    plt.xlabel(feature)
    plt.ylabel(data_label1.columns[-1])
    plt.show()


# TPV
for feature in data_label2.columns[:-1]:  # 最后一列是预测目标，不需要绘制
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=feature, y=data_label2.columns[-1], data=data_label2)
    plt.xlabel(feature)
    plt.ylabel(data_label2.columns[-1])
    plt.show()


#MIC
for feature in data_label3.columns[:-1]:  # 最后一列是预测目标，不需要绘制
    plt.figure(figsize=(6, 6))
    sns.lineplot(x=feature, y=data_label3.columns[-1], data=data_label3)
    plt.xlabel(feature)
    plt.ylabel(data_label3.columns[-1])
    plt.show()
'''

