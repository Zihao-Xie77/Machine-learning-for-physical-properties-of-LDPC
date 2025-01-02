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

# Importation of the cleaned dataset
date = pd.read_csv('Machine learning for physical properties of LDPC/Dataset/Cleaned Dataset')

s = StratifiedShuffleSplit(n_splits=1, train_size=0.8, test_size=0.2, random_state=77) # random seed for dataset separation
strata_variable = date[['Agent_H3PO4', 'Agent_K2CO3', 'Agent_KOH', 'Agent_Na2CO3', 'Agent_NaOH',
                        'Agent_ZnCl2']]
for train_index, test_index in s.split(date, strata_variable):
    # print(train_index,test_index)
    strat_train_set = date.loc[train_index]
    strat_test_set = date.loc[test_index]

my_pipeline = Pipeline([
    ('std_scaler', StandardScaler())    
])

# Predicting features selection
features_train = strat_train_set.drop(['SBET (cm2/g)', 'TPV (cm3/g)', 'Microporous_Ratio (%)'], axis=1)
label1_train = strat_train_set['SBET (cm2/g)']
label2_train = strat_train_set['TPV (cm3/g)']
label3_train = strat_train_set['Microporous_Ratio (%)']

features_test = strat_test_set.drop(['SBET (cm2/g)', 'TPV (cm3/g)', 'Microporous_Ratio (%)'], axis=1)
label1_test = strat_test_set['SBET (cm2/g)']
label2_test = strat_test_set['TPV (cm3/g)']
label3_test = strat_test_set['Microporous_Ratio (%)']

features_prepared_train = my_pipeline.fit_transform(features_train)
features_prepared_test = my_pipeline.fit_transform(features_test)


# RF_based GBR models for SSA, TPV, and MP
# random seeds: dataset separation (77);  model (77)
model1GBR_416 = GradientBoostingRegressor(n_estimators=170, learning_rate=0.2, max_depth=6, max_features=6,
                                          min_samples_split=5, min_samples_leaf=4, random_state=77) # SSA
# random seeds: dataset separation (77);  model (78)
model2GBR_416 = GradientBoostingRegressor(n_estimators=170, learning_rate=0.56, max_depth=7, max_features=7,
                                          min_samples_split=4, min_samples_leaf=6, random_state=78) # TPV
# random seeds: dataset separation (79);  model (77)
model3GBR_416 = GradientBoostingRegressor(n_estimators=160, learning_rate=0.2, max_depth=6, max_features=4,
                                          min_samples_split=5, min_samples_leaf=4, random_state=77) # MP

model1GBR_416.fit(features_prepared_train, label1_train)
model2GBR_416.fit(features_prepared_train, label2_train)
model3GBR_416.fit(features_prepared_train, label3_train)


# region Evaluation of the final regressor
class Evaluate:
    def __init__(self, model, features, labels):
        self.model = model
        self.features = features
        self.labels = labels

# RMSE
    def rmse(self):
        from sklearn.metrics import mean_squared_error
        prediction = self.model.predict(self.features)
        mse = mean_squared_error(self.labels, prediction)
        rmse = np.sqrt(mse)
        print("-- RMSE --")
        print(f"Root Mean Square Error is: {rmse}")
# Mean and Standard deviation
    def cross_validation(self):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.features, self.labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        print("-- Cross Validation --")
        print("Mean: ", rmse_scores.mean())
        print("Standard deviation: ", rmse_scores.std())
# R2
    def r2_score(self):
        from sklearn.metrics import r2_score
        label_prediction = self.model.predict(self.features)
        r2 = r2_score(self.labels, label_prediction)
        print("-- R2 Score --")
        print("R2: ", r2)
# endregion


# region Print the evalutation
print("\n\n--- SSA ----")

print("\n\n-- TRAIN EVALUATION ---")
e1 = Evaluate(model1GBR_416, features_prepared_train, label1_train)
e1.rmse()
e1.cross_validation()
e1.r2_score()

print("\n\n--- TEST EVALUATION ---")
e2 = Evaluate(model1GBR_416, features_prepared_test, label1_test)
e2.rmse()
e2.r2_score()

print("\n\n--- TPV ----")

print("\n\n-- TRAIN EVALUATION ---")
e3 = Evaluate(model2GBR_416, features_prepared_train, label2_train)
e3.rmse()
e3.cross_validation()
e3.r2_score()

print("\n\n--- TEST EVALUATION ---")
e4 = Evaluate(model2GBR_416, features_prepared_test, label2_test)
e4.rmse()
e4.r2_score()

print("\n\n--- MP ----")

print("\n\n-- TRAIN EVALUATION ---")
e5 = Evaluate(model3GBR_416, features_prepared_train, label3_train)
e5.rmse()
e5.cross_validation()
e5.r2_score()

print("\n\n--- TEST EVALUATION ---")
e6 = Evaluate(model3GBR_416, features_prepared_test, label3_test)
e6.rmse()
e6.r2_score()
print("----- Program Finished -----")
# endregion


# region Partial dependence of the final regressor
# Input feature index
feature_index = 11
fig, ax = plt.subplots(figsize=(8.66, 6.93), dpi=500)
display = PartialDependenceDisplay.from_estimator(model3GBR_416, features_prepared_H3PO4, [feature_index],
                                                  kind='average', ax=ax,
                                                  line_kw={"color": "green"},
                                                  pd_line_kw={"alpha": 1.0})
plt.show()
pdp_results = display.pd_results[0]
pdp_values = pdp_results['grid_values'][0]  
pdp_avg = pdp_results['average'] 

if pdp_avg.ndim > 1:
    pdp_avg = pdp_avg.flatten()
frac = 0  
smoothed_pdp_avg = lowess(pdp_avg, pdp_values, frac=frac)[:, 1]
pdp_df = pd.DataFrame({
    'Feature Value': pdp_values,
    'Partial Dependence': smoothed_pdp_avg
})

pdp_df.to_csv('PD_H3PO4_MP.csv', index=False)
# endregion


predictions_model1 = model1GBR_416.predict(features_prepared_train)
predictions_model2 = model2GBR_416.predict(features_prepared_train)
predictions_model3 = model3GBR_416.predict(features_prepared_train)

predictions_test_model1 = model1GBR_416.predict(features_prepared_test)
predictions_test_model2 = model2GBR_416.predict(features_prepared_test)
predictions_test_model3 = model3GBR_416.predict(features_prepared_test)


# region Pearson correlation plot
plt.figure(figsize=(20, 20))
sns.set(font_scale=1.2)

cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
            square=True, cbar_kws={"shrink": 0.75}, vmin=-1, vmax=1, linewidths=0.5)

plt.tight_layout()
plt.show()
# endregion


# region Scatter plots of the prediction of the final regressor
# SBET
model = LinearRegression()
model.fit(predictions_model1.reshape(-1, 1), label1_train)
plt.figure(figsize=(8.66, 8.66), dpi=500)

plt.scatter(predictions_model1, label1_train)
plt.scatter(predictions_test_model1, label1_test)

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

font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

font_properties = FontProperties(fname=font_path, size=15, weight='bold')

plt.legend(prop=font_properties, loc='upper left')
plt.tight_layout()
plt.show()


# TPV
model = LinearRegression()
model.fit(predictions_model2.reshape(-1, 1), label2_train)
plt.figure(figsize=(8.66, 8.66), dpi=500)

plt.scatter(predictions_model2, label2_train)
plt.scatter(predictions_test_model2, label2_test)

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

font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

font_properties = FontProperties(fname=font_path, size=15, weight='bold')

plt.legend(prop=font_properties, loc='upper left')
plt.tight_layout()
plt.show()


# MP
model = LinearRegression()
model.fit(predictions_model3.reshape(-1, 1), label3_train)
plt.figure(figsize=(8.66, 8.66), dpi=500)

plt.scatter(predictions_model3, label3_train)
plt.scatter(predictions_test_model3, label3_test)

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

font_path = fm.findfont(fm.FontProperties(family='Arial'))
print(font_path)

font_properties = FontProperties(fname=font_path, size=15, weight='bold')

plt.legend(prop=font_properties, loc='upper left')
plt.tight_layout()
plt.show()
# endregion


# region Output of the Scatter plots 
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
df.to_csv('PA-GBR.csv', index=False)
# endregin


# region Feature importance analysis
class FeatureImportancePlotter:
    def __init__(self, feature_importances, custom_labels):
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
        group_importances = {}
        for group, features in self.groups.items():
            group_importance = sum(self.feature_importances[self.custom_labels.index(f)] for f in features)
            group_importances[group] = group_importance
        return group_importances

    def plot_group_importances(self):
        group_importances = self.calculate_group_importances()
        labels = list(group_importances.keys())
        sizes = list(group_importances.values())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        plt.figure(figsize=(8, 8), dpi=500)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
        plt.axis('equal') 
        plt.title('Grouped Feature Importance', fontproperties='Arial', fontsize=15, fontweight='bold')
        plt.show()
# endregion

# region feature importance plots
# SSA
feature_importances = (model1GBR_416.feature_importances_)

custom_labels1 = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

dd1 = pd.DataFrame({
    'Feature': custom_labels1,
    'ImportanceSSA': feature_importances
})

dd1.to_csv('feature_importances.csv', index=False)

interested_features = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
interested_indices = [custom_labels1.index(feature) for feature in interested_features]
interested_importances = feature_importances[interested_indices]
custom_labels2 = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

sorted_indices = np.argsort(interested_importances)
sorted_importances = interested_importances[sorted_indices]
sorted_labels = [custom_labels2[i] for i in sorted_indices]

plt.figure(figsize=(8.66, 6.93), dpi=500)
bars = plt.barh(range(len(interested_importances)), sorted_importances, align='center', color='darkturquoise')
plt.xlabel('Feature Importance of SSA', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=6) 
plt.xticks(x_ticks, fontsize=15)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  

plt.tight_layout()
plt.show()

plotter = FeatureImportancePlotter(feature_importances, custom_labels1)
plotter.plot_group_importances()

# TPV
feature_importances = (model2GBR_416.feature_importances_)

custom_labels = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
dd2 =pd.read_csv('feature_importances.csv')
dd2['importanceTPV'] = feature_importances
dd2.to_csv('feature_importances.csv', index=False)
interested_features = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
interested_indices = [custom_labels.index(feature) for feature in interested_features]
interested_importances = feature_importances[interested_indices]

custom_labels = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
sorted_indices = np.argsort(interested_importances)
sorted_importances = interested_importances[sorted_indices]
sorted_labels = [custom_labels[i] for i in sorted_indices]

plt.figure(figsize=(8.66, 6.93), dpi=500)
bars = plt.barh(range(len(interested_importances)), sorted_importances, align='center', color='orange')
plt.xlabel('Feature Importance of TPV', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=6)  
plt.xticks(x_ticks, fontsize=15)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  

plt.tight_layout()
plt.show()

# MP
feature_importances = (model3GBR_416.feature_importances_)

custom_labels = ['C', 'H', 'O', 'N', 'S', 'VM', 'Ash', 'FC',
                 'A/S', 'HR', 'Temp', 'Time',
                 '$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
dd3 =pd.read_csv('feature_importances.csv')
dd3['importanceMP'] = feature_importances
dd3.to_csv('feature_importances.csv', index=False)

interested_features = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']
interested_indices = [custom_labels.index(feature) for feature in interested_features]
interested_importances = feature_importances[interested_indices]

custom_labels = ['$H_{\mathrm{3}}$P$O_{\mathrm{4}}$',
                 '$K_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'KOH',
                 '$Na_{\mathrm{2}}$C$O_{\mathrm{3}}$', 'NaOH', 'Zn$Cl_{\mathrm{2}}$']

sorted_indices = np.argsort(interested_importances)
sorted_importances = interested_importances[sorted_indices]
sorted_labels = [custom_labels[i] for i in sorted_indices]

plt.figure(figsize=(8.66, 6.93), dpi=500)
bars = plt.barh(range(len(interested_importances)), sorted_importances, align='center', color='sandybrown')
plt.xlabel('Feature Importance of MP', fontproperties='Arial', fontsize=15, fontweight='bold')
plt.yticks(range(len(sorted_importances)), sorted_labels, fontproperties='Arial', fontsize=15)
x_ticks = np.linspace(0, 0.25, num=6) 
plt.xticks(x_ticks, fontsize=15)
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  
plt.tight_layout()
plt.show()
# endregion


# region Feature importance analysis under KOH 
KOH_subset = date[date['Agent_KOH'] == 1]
features_KOH = KOH_subset.drop(['SBET (cm2/g)', 'TPV (cm3/g)', 'Microporous_Ratio (%)'], axis=1)
label1_KOH = KOH_subset['SBET (cm2/g)']
label2_KOH = KOH_subset['TPV (cm3/g)']
features_prepared_KOH = my_pipeline.transform(features_KOH)
result1 = permutation_importance(model1GBR_416, features_prepared_KOH, label1_KOH, n_repeats=10, random_state=42)
result2 = permutation_importance(model2GBR_416, features_prepared_KOH, label2_KOH, n_repeats=10, random_state=42)
feature_names = features_KOH.columns
feature_importance_df1 = pd.DataFrame({'Feature': feature_names, 'Importance': result1.importances_mean})
feature_importance_df2 = pd.DataFrame({'Feature': feature_names, 'Importance': result2.importances_mean})

feature_importance_df1.to_csv('fi_KOH_SSA.csv', index=False)
feature_importance_df2.to_csv('fi_KOH_TPV.csv', index=False)
# endregion


# region Feature importance analysis under H3PO4
H3PO4_subset = date[date['Agent_H3PO4'] == 1]
features_H3PO4 = H3PO4_subset.drop(['SBET (cm2/g)', 'TPV (cm3/g)', 'Microporous_Ratio (%)'], axis=1)
label3_H3PO4 = H3PO4_subset['Microporous_Ratio (%)']
features_prepared_H3PO4 = my_pipeline.transform(features_H3PO4)
result3 = permutation_importance(model3GBR_416, features_prepared_H3PO4, label3_H3PO4, n_repeats=10, random_state=42)
feature_names = features_H3PO4.columns
feature_importance_df1 = pd.DataFrame({'Feature': feature_names, 'Importance': result3.importances_mean})

feature_importance_df1.to_csv('fi_H3PO4_MP.csv', index=False)
# endregion


# region Partial dependence of the final regressor
# Input feature index
feature_index = 11
fig, ax = plt.subplots(figsize=(8.66, 6.93), dpi=500)
display = PartialDependenceDisplay.from_estimator(model3GBR_416, features_prepared_H3PO4, [feature_index],
                                                  kind='average', ax=ax,
                                                  line_kw={"color": "green"},
                                                  pd_line_kw={"alpha": 1.0})
plt.show()
pdp_results = display.pd_results[0]
pdp_values = pdp_results['grid_values'][0]  
pdp_avg = pdp_results['average'] 

if pdp_avg.ndim > 1:
    pdp_avg = pdp_avg.flatten()
frac = 0  
smoothed_pdp_avg = lowess(pdp_avg, pdp_values, frac=frac)[:, 1]
pdp_df = pd.DataFrame({
    'Feature Value': pdp_values,
    'Partial Dependence': smoothed_pdp_avg
})

pdp_df.to_csv('PD_H3PO4_MP.csv', index=False)
# endregion

