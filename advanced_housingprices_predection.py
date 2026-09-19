import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from regex import R, T
import kagglehub
import os
from sklearn.base import clone
from torch import nn

####loading data
df = pd.read_csv('C:\\Users\\user\\Downloads\\train_kaggle.csv')
df_test = pd.read_csv('C:\\Users\\user\\Downloads\\test_kaggle.csv')
test_id = df_test['Id']
####data understanding and exploration (train)
print(df.columns)
print(f'shape', df.shape)
print(df.info())
print(df.describe().T)
print(f'duplicates', df.duplicated().sum())
columns_with_missings = df.columns.where(df.isnull().sum()/len(df) > 0).dropna().to_list()
print(f'count of columns with missings', len(columns_with_missings))
print(f'missing values percentage', columns_with_missings)
missings= (df[columns_with_missings].isnull().sum()/len(df)*100).sort_values(ascending=False , inplace=False)
print(f'missing %', missings) 
print(f'missing values types',df[missings.index].info() )
print( df[missings.index].describe().T)
categorical_missings = (df[missings.index].select_dtypes(exclude=['float64', 'int64']))
for col in categorical_missings.columns:
    print(f'missings in {col}', categorical_missings[col].value_counts(dropna = False ,normalize=True)*100)

# data understanding and exploration (test)
print(df_test.columns)
print(f'shape test', df_test.shape)
print(df_test.info())
print(df_test.describe().T)
print(f'duplicates', df_test.duplicated().sum())
columns_with_missings_test = df_test.columns.where(df_test.isnull().sum()/len(df_test) > 0).dropna().to_list()
print(f'count of columns with missings', len(columns_with_missings_test))
print(f'missing values percentage', columns_with_missings_test)
missings_test= (df_test[columns_with_missings_test].isnull().sum()/len(df_test)*100).sort_values(ascending=False , inplace=False)
print(f'missing %', missings_test)

# Global missings imputation

No_pool = 0
No_response = 0
for i in range(df.shape[0]):
    if df.loc[i, 'PoolQC'] is np.nan and df.loc[i, 'PoolArea'] == 0:
        No_pool += 1
    elif df.loc[i, 'PoolQC'] is np.nan and df.loc[i, 'PoolArea'] > 0:
        No_response += 1
print(f'No pool: {No_pool}')
print(f'No response: {No_response}')
        

df_clean = df.copy()
df_clean['PoolQC'] = df_clean['PoolQC'].fillna('No Pool')
df_test['PoolQC'] = df_test['PoolQC'].fillna('No Pool')
df_clean = df_clean.drop(columns=['PoolArea'])
df_test = df_test.drop(columns=['PoolArea'])

No_misc = 0
No_response = 0
for i in range(df_clean.shape[0]):
    if df_clean.loc[i,'MiscFeature'] is np.nan and df_clean.loc[i,'MiscVal'] == 0:
        No_misc+= 1
    elif df_clean.loc[i,'MiscFeature'] is np.nan and df_clean.loc[i,'MiscVal'] > 0:
        No_response += 1
print(f'No misc: {No_misc}')
print(f'No response: {No_response}')
df_clean['MiscFeature'] = df_clean['MiscFeature'].fillna('No Misc')
df_test['MiscFeature'] = df_test['MiscFeature'].fillna('No Misc')
df_clean['Alley'] = df_clean['Alley'].fillna('Not in Alley')
df_test['Alley'] = df_test['Alley'].fillna('Not in Alley')
df_clean['Fence'] = df_clean['Fence'].fillna('No Fence')
df_test['Fence'] = df_test['Fence'].fillna('No Fence')



## garage missings
no_garage = 0
no_response = 0
for i in range(df_clean.shape[0]):
    if pd.isnull(df_clean.loc[i, 'GarageYrBlt']):
        if pd.isnull(df_clean.loc[i, 'GarageFinish']):
            if pd.isnull(df_clean.loc[i, 'GarageQual']):
                if pd.isnull(df_clean.loc[i, 'GarageCond']):
                    no_garage += 1
    else:
        no_response += 1
print(f'no_garage', no_garage)
print(f'no_response', no_response)
print((no_garage/df_clean.shape[0])*100)
        


df_clean[['GarageType'  , 'GarageFinish' , 'GarageQual' , 'GarageCond']]  = df_clean[['GarageType'  , 'GarageFinish' , 'GarageQual' , 'GarageCond']].fillna('No Garage')
print(f'GarageYrBlt data type', df_clean["GarageYrBlt"].dtype)
df_clean['GarageYrBlt'] = df_clean['GarageYrBlt'].fillna(0)
# df_clean['has_garage'] = df_clean['GarageType'].map(lambda x: 1 if x != 'No Garage' else 0)
# df_clean['has_garage'] = df_clean['has_garage'].astype('object')
df_test[['GarageType'  , 'GarageFinish' , 'GarageQual' , 'GarageCond']]  = df_test[['GarageType'  , 'GarageFinish' , 'GarageQual' , 'GarageCond']].fillna('No Garage')
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(0)
# df_test['has_garage'] = df_test['GarageType'].map(lambda x: 1 if x != 'No Garage' else 0)
# df_test['has_garage'] = df_test['has_garage'].astype('object')



## basment missings
no_basement = 0
no_response = 0
for i in range(df_clean.shape[0]):
    if pd.isnull(df_clean.loc[i, 'BsmtExposure']):
        if pd.isnull(df_clean.loc[i, 'BsmtFinType2']):
            if pd.isnull(df_clean.loc[i, 'BsmtQual']):
                if pd.isnull(df_clean.loc[i, 'BsmtFinType1']):
                    no_basement += 1
    else:
        no_response += 1
print(f'no_basement', no_basement)
print(f'no_response', no_response)
print((no_basement/df_clean.shape[0])*100)
        
df_clean[['BsmtExposure' , 'BsmtFinType2' , 'BsmtQual' , 'BsmtFinType1' , 'BsmtCond']]  = df_clean[['BsmtExposure' , 'BsmtFinType2' , 'BsmtQual' , 'BsmtFinType1' , 'BsmtCond']].fillna('No Basement')
df_test[['BsmtExposure' , 'BsmtFinType2' , 'BsmtQual' , 'BsmtFinType1' , 'BsmtCond']]  = df_test[['BsmtExposure' , 'BsmtFinType2' , 'BsmtQual' , 'BsmtFinType1' , 'BsmtCond']].fillna('No Basement')
df_clean['Electrical'] = df_clean['Electrical'].fillna(df_clean['Electrical'].mode()[0])
df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])


df_clean['DateSold'] = pd.to_datetime(df_clean['YrSold'].astype(str) + '-' + df_clean['MoSold'].astype(str) + '-01')
df_clean['YrSold'] = df_clean['YrSold'].astype(int)
df_test['DateSold'] = pd.to_datetime(df_test['YrSold'].astype(str) + '-' + df_test['MoSold'].astype(str) + '-01')
df_test['YrSold'] = df_test['YrSold'].astype(int)
#gonna leave the month as seasonality detection
df_clean['MoSold'] = df_clean['MoSold'].astype(str)
df_test['MoSold'] = df_test['MoSold'].astype(str)

df_clean['House_age_at_sale'] = df_clean['YrSold'] - df_clean['YearBuilt']
df_test['House_age_at_sale'] = df_test['YrSold'] - df_test['YearBuilt']


df_test['BsmtHalfBath'] = df_test['BsmtHalfBath'].fillna(df_clean['BsmtHalfBath'].mode()[0])
df_clean['BsmtBath'] =df_clean['BsmtHalfBath'].map(lambda x: 'no' if x == 0 else 'yes')
df_clean = df_clean.drop(columns=['BsmtHalfBath'])
df_clean['BsmtBath'] = df_clean['BsmtBath'].astype('object')
df_test['BsmtBath'] =df_test['BsmtHalfBath'].map(lambda x: 'no' if x == 0 else 'yes')
df_test = df_test.drop(columns=['BsmtHalfBath'])
df_test['BsmtBath'] = df_test['BsmtBath'].astype('object')

zero_dominant_features = []

for col in df_clean.columns:
        if (df_clean[col].dtype == 'int64' or df_clean[col].dtype == 'float64') and df_clean[col].quantile(0.95) == 0:
            zero_dominant_features.append(col)
print('zero dominant features', zero_dominant_features)
print(f'zero dominant features count', len(zero_dominant_features))
df_clean = df_clean.drop(columns=zero_dominant_features)
df_test = df_test.drop(columns=zero_dominant_features)
        

# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
# # train-valid split
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
df_train , df_test_train = train_test_split(df_clean, test_size=0.12, random_state=42 )
df_train_train , df_valid_train = train_test_split(df_train, test_size=0.1, random_state=42 )




print(f'final train set description', df_train_train.describe().T)
print('final train set info', df_train_train.info())
print(f'final train set missings', (df_train_train.isna().sum()).sort_values(ascending=False))





# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
# ##the rest of the missings imputation

no_masonry = 0
no_response = 0
for i in range(df_clean.shape[0]):
    if df_clean.loc[i, 'MasVnrType'] is np.nan and (df_clean.loc[i, 'MasVnrArea'] == 0 or df_clean.loc[i, 'MasVnrArea'] is np.nan):
        no_masonry += 1
    elif df_clean.loc[i, 'MasVnrType'] is np.nan and df_clean.loc[i, 'MasVnrArea'] > 0:
        no_response += 1
print(f'no_masonry', no_masonry)
print(f'no_response', no_response)
print((no_masonry/df_clean.shape[0])*100)


MasVnrArea_median = df_train_train["MasVnrArea"].median()
df_train_train["MasVnrArea"] = df_train_train["MasVnrArea"].fillna(MasVnrArea_median)
df_valid_train["MasVnrArea"] = df_valid_train["MasVnrArea"].fillna(MasVnrArea_median)
df_test_train["MasVnrArea"] = df_test_train["MasVnrArea"].fillna(MasVnrArea_median)
df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(MasVnrArea_median)

train_masvnr_mode = df_train_train['MasVnrType'].mode()[0]


for df in [df_train_train, df_valid_train, df_test_train , df_test]:
  
    df.loc[df['MasVnrType'].isna() & (df['MasVnrArea'] == 0), 'MasVnrType'] = 'No Masonry'
    df.loc[df['MasVnrType'].isna() & (df['MasVnrArea'] > 0), 'MasVnrType'] = train_masvnr_mode



LotFrontage_median = df_train_train["LotFrontage"].median()
df_train_train["LotFrontage"] = df_train_train["LotFrontage"].fillna(LotFrontage_median)
df_valid_train["LotFrontage"] = df_valid_train["LotFrontage"].fillna(LotFrontage_median)
df_test_train["LotFrontage"] = df_test_train["LotFrontage"].fillna(LotFrontage_median)
df_test["LotFrontage"] = df_test["LotFrontage"].fillna(LotFrontage_median)



#gonna try again with deleting that feature to not to bias my analysis
probs_fireplace = df_train_train['FireplaceQu'].value_counts(normalize=True)
null_value_fireplace = df_train_train['FireplaceQu'].isna()
null_sum_fireplace = null_value_fireplace.sum()
df_train_train.loc[null_value_fireplace, 'FireplaceQu'] = np.random.choice(
    a=probs_fireplace.index,
    size=null_sum_fireplace,
    p=probs_fireplace.values
)


probs_fireplace_valid = df_valid_train['FireplaceQu'].value_counts(normalize=True)
null_value_fireplace_valid = df_valid_train['FireplaceQu'].isna()
null_sum_fireplace_valid = null_value_fireplace_valid.sum()
df_valid_train.loc[null_value_fireplace_valid, 'FireplaceQu'] = np.random.choice(
    a=probs_fireplace_valid.index,
    size=null_sum_fireplace_valid,
    p=probs_fireplace_valid.values
)

probs_fireplace_test_train = df_test_train['FireplaceQu'].value_counts(normalize=True)
null_value_fireplace_test_train = df_test_train['FireplaceQu'].isna()
null_sum_fireplace_test_train = null_value_fireplace_test_train.sum()
df_test_train.loc[null_value_fireplace_test_train, 'FireplaceQu'] = np.random.choice(
    a=probs_fireplace_test_train.index,
    size=null_sum_fireplace_test_train,
    p=probs_fireplace_test_train.values
)

test_probs_fireplace = df_test['FireplaceQu'].value_counts(normalize=True)
null_value_fireplace_test = df_test['FireplaceQu'].isna()
null_sum_fireplace_test = null_value_fireplace_test.sum()
df_test.loc[null_value_fireplace_test, 'FireplaceQu'] = np.random.choice(
    a=test_probs_fireplace.index,
    size=null_sum_fireplace_test,
    p=test_probs_fireplace.values
)
print(f'class distribution after imputation (train)', df_train_train['FireplaceQu'].value_counts(normalize=True))
print(f'class distribution after imputation (valid)', df_valid_train['FireplaceQu'].value_counts(normalize=True))
print(f'class distribution after imputation (test)', df_test_train['FireplaceQu'].value_counts(normalize=True))
print(f'class distribution after imputation (test)', df_test['FireplaceQu'].value_counts(normalize=True))

Quality_Features= df_train_train[['OverallQual' , 'BsmtFinType1' , 'BsmtFinType2' , 'HeatingQC'  , 'KitchenQual' , 'FireplaceQu' , 'GarageQual' , 'PoolQC'  ,'Fence']]
for col in Quality_Features.columns:
    print (f'{col} count: {Quality_Features[col].value_counts(normalize=True)*100}')
df_train_train[Quality_Features.columns] = df_train_train[Quality_Features.columns].astype('object')
df_valid_train[Quality_Features.columns] = df_valid_train[Quality_Features.columns].astype('object')
df_test_train[Quality_Features.columns] = df_test_train[Quality_Features.columns].astype('object')
df_test[Quality_Features.columns] = df_test[Quality_Features.columns].astype('object')


Condition_Features= df_train_train[['OverallCond' , 'Condition1' , 'Condition2' , 'BsmtCond' , 'GarageCond' , 'SaleCondition']]
for col in Condition_Features.columns:
    print (f'{col} count: {Condition_Features[col].value_counts(normalize=True)*100}')
df_train_train[Condition_Features.columns] = df_train_train[Condition_Features.columns].astype('object')
df_valid_train[Condition_Features.columns] = df_valid_train[Condition_Features.columns].astype('object')
df_test_train[Condition_Features.columns] = df_test_train[Condition_Features.columns].astype('object')
df_test[Condition_Features.columns] = df_test[Condition_Features.columns].astype('object')

missing_cols_test = df_test.columns.where(df_test.isnull().sum()/len(df_test) > 0).dropna().to_list()
for col in missing_cols_test:
    if df_test[col].dtype == 'int64' or df_test[col].dtype == 'float64':
        
        df_test[col] = df_test[col].fillna(df_train[col].median())
    else:
        df_test[col] = df_test[col].fillna(df_train[col].mode()[0])

print(f' train set missings', (df_train_train.isna().sum()).sort_values(ascending=False))
print(f' valid set missings', (df_valid_train.isna().sum()).sort_values(ascending=False))
print(f' test set missings', (df_test_train.isna().sum()).sort_values(ascending=False))
print(f' test set missings', (df_test.isna().sum()).sort_values(ascending=False))








# ################################################################################################################################################################################################################################################################
# #################################################################################################################################################################################################################################################################
## feature engineering 
# feature transformation
print(f'final train set description', df_train_train.describe().T)
print('final train set info', df_train_train.info())
print(f'final train set missings', (df_train_train.isna().sum()).sort_values(ascending=False))



x_train = df_train_train.drop(columns=['SalePrice' , 'Id'])
x_valid = df_valid_train.drop(columns=['SalePrice' , 'Id'])
x_test_train = df_test_train.drop(columns=[ 'SalePrice' , 'Id'])
x_test = df_test.drop(columns=['Id'])
y_train = df_train_train['SalePrice']
y_valid = df_valid_train['SalePrice']
y_test_train = df_test_train['SalePrice']

x_train_numeric = x_train.select_dtypes(include=['int64', 'float64'])
x_valid_numeric = x_valid.select_dtypes(include=['int64', 'float64'])
x_test_train_numeric = x_test_train.select_dtypes(include=['int64', 'float64'])
x_test_numeric = x_test.select_dtypes(include=['int64', 'float64'])
x_train_categorical = x_train.select_dtypes(exclude=['int64', 'float64'])
x_valid_categorical = x_valid.select_dtypes(exclude=['int64', 'float64'])
x_test_train_categorical = x_test_train.select_dtypes(exclude=['int64', 'float64'])
x_test_categorical = x_test.select_dtypes(exclude=['int64', 'float64'])
#adding dummies to the numerical variables where the feature might not be included 


#distinguishing between discrete and continuous features
x_discrete = []
x_continuous = []
year_features = ['GarageYrBlt' , 'YearBuilt' ,'YearRemodAdd' , 'YrSold' , 'House_age_at_sale']
for col in x_train_numeric.columns:
    if x_train_numeric[col].nunique() <= 15 or col in year_features  :
        x_discrete.append(col)
    else:
        x_continuous.append(col)
 
x_train_discrete = x_train_numeric[x_discrete]
x_train_continuous = x_train_numeric[x_continuous]
x_valid_discrete = x_valid_numeric[x_discrete]
x_valid_continuous = x_valid_numeric[x_continuous]
x_test_train_discrete = x_test_train_numeric[x_discrete]
x_test_train_continuous = x_test_train_numeric[x_continuous]
x_test_discrete = x_test_numeric[x_discrete]
x_test_continuous = x_test_numeric[x_continuous]
        

dummies_cols = []
cols_to_dummies = []
for df in [x_train_numeric, x_valid_numeric,x_test_train_numeric, x_test_numeric]:
    for col in df.columns:
        if x_train_numeric[col].median() == 0:
            # df[f'has_{col}'] = df[col].map(lambda x: 'no' if x == 0 else 'yes')
            # df[f'has_{col}'] = df[f'has_{col}'].astype('object')
            dummies_cols.append(f'has_{col}')
            cols_to_dummies.append(col)
            
    print(f'dummies_cols for {df} are {dummies_cols} and there are {len(dummies_cols)} dummies')
    initial_zero_driven_features = (set(cols_to_dummies).intersection(x_continuous))
    print(f'dummies_cols in countinous features for {df} are {set(cols_to_dummies).intersection(x_continuous)} and there quantity is {len(set(cols_to_dummies).intersection(x_continuous))}')
    dummies_cols = []
    cols_to_dummies = []
zero_driven_features_cols = list(dict.fromkeys(initial_zero_driven_features))
print(f'zero_driven_features_cols {zero_driven_features_cols}')








#outliers detection
outliers_features = []
for col in x_train_continuous.columns:
    q1 = x_train_continuous[col].quantile(0.25)
    q3 = x_train_continuous[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    outliers = (x_train_continuous[col][(x_train_continuous[col] > upper_limit ) | (x_train_continuous[col] < lower_limit )]).count()
    #####depicting whether it's an outlier or leverage
    if outliers > 0:
        print(f'Outliers in {col}: {outliers}')
        outliers_features.append(col)
        plt.scatter(x_train_continuous[col], y_train)
        plt.title(f'Outliers in {col}')
        # plt.show()
        #outta try whether we need to treat MasVnrArea outliers or to preserve them as leverages
        #outta include the garage and masonry dummies if'll include their countinuos features
    plt.boxplot(x_train_continuous[col])
    plt.title(f'Boxplot of {col}')
    # plt.show()
print(f'outliers features count {len(outliers_features)}')
zero_driven_features = x_train_continuous[zero_driven_features_cols][x_train_continuous[zero_driven_features_cols]!=0]
outliers_features = [col for col in outliers_features if col not in zero_driven_features.columns]
for df in [x_train_continuous, x_valid_continuous , x_test_train_continuous, x_test_continuous]:
    for col in df[outliers_features].columns:
        upper_limit = x_train_continuous[col].quantile(0.95)
        lower_limit = x_train_continuous[col].quantile(0.05)
        df[col] = df[col].clip( lower=lower_limit, upper=upper_limit)
        
        
        
        
print('####################################################################################################################################################################################################')
#outliers detection
outliers_features_after_capping = []
for col in x_train_continuous.columns:
    q1 = x_train_continuous[col].quantile(0.25)
    q3 = x_train_continuous[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    outliers = (x_train_continuous[col][(x_train_continuous[col] > upper_limit ) | (x_train_continuous[col] < lower_limit )]).count()
    #####depicting whether it's an outlier or leverage
    if outliers > 0:
        print(f'Outliers after capping in {col}: {outliers}')
        outliers_features_after_capping.append(col)
        plt.scatter(x_train_continuous[col], y_train)
        plt.title(f'Outliers in {col}')
        # plt.show()
        #outta try whether we need to treat MasVnrArea outliers or to preserve them as leverages
        #outta include the garage and masonry dummies if'll include their countinuos features
    plt.boxplot(x_train_continuous[col])
    plt.title(f'Boxplot of {col}')
    # plt.show()
print(f'outliers features after capping count {len(outliers_features_after_capping)}')
print('####################################################################################################################################################################################################')
 ########################################################################################################################################################################################################
### mitigating the impact of the zero on these features and its tendency to flag unrealistic outliers
zero_outliers_features = []
for col in zero_driven_features.columns:
    q1 = zero_driven_features[col].quantile(0.25)
    q3 = zero_driven_features[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    outliers = (zero_driven_features[col][(zero_driven_features[col] > upper_limit ) | (zero_driven_features[col] < lower_limit )]).count()
    #####depicting whether it's an outlier or leverage
    if outliers > 0:
        print(f'Outliers for zero driven in {col}: {outliers}')
        zero_outliers_features.append(col)
        plt.scatter(zero_driven_features[col], y_train)
        plt.title(f'Outliers in {col}')
        # plt.show()
        
        
        #outta try whether we need to treat MasVnrArea outliers or to preserve them as leverages
        #outta include the garage and masonry dummies if'll include their countinuos features
    
    plt.boxplot(zero_driven_features[col])
    plt.title(f'Boxplot of zero driven features {col}')
    # plt.show()
    
print(f'zero_outliers_features count {len(zero_outliers_features)}')
to_remove = ['2ndFlrSF']
for col in to_remove:
    zero_outliers_features.remove(col)
     
    

#perhabs they're leverages not outliers
                              
for df in [x_train_continuous, x_valid_continuous, x_test_train_continuous, x_test_continuous]:
    for col in df[zero_outliers_features].columns:
        upper_limit = zero_driven_features[col].quantile(0.95)
        lower_limit = zero_driven_features[col].quantile(0.05)
        df[col] = df[col].clip( lower=lower_limit, upper=upper_limit)
        
print('####################################################################################################################################################################################################')
    
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
#outliers detection
outliers_features_final = []
for col in x_train_continuous.columns:
    q1 = x_train_continuous[col].quantile(0.25)
    q3 = x_train_continuous[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (1.5 * iqr)
    upper_limit = q3 + (1.5 * iqr)
    outliers_final = (x_train_continuous[col][(x_train_continuous[col] > upper_limit ) | (x_train_continuous[col] < lower_limit )]).count()
    #####depicting whether it's an outlier or leverage
    if outliers_final > 0:
        print(f' final Outliers in {col}: {outliers_final}')
        outliers_features_final.append(col)
        plt.scatter(x_train_continuous[col], y_train)
        plt.title(f'Outliers in {col}')
        # plt.show()
        #outta try whether we need to treat MasVnrArea outliers or to preserve them as leverages
        #outta include the garage and masonry dummies if'll include their countinuos features
    plt.boxplot(x_train_continuous[col])
    plt.title(f'Boxplot of {col}')
    # plt.show()
print(f'outliers features final count {len(outliers_features_final)}')
    
    
    
# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
#skewness


flagged_skewness_train = []

print (f'Skewness flagged before transforming')
for col in x_train_continuous.columns:
    skewness = x_train_continuous[col].skew().round(2)
    if skewness > 1 or skewness <= -1:
        print(f'{col} is highly skewed with a skewness of {skewness:.2f}')
        flagged_skewness_train.append(col)
    elif skewness > 0.5 or skewness < -0.5:
        print(f'{col} is moderately skewed with a skewness of {skewness:.2f}')
    else:
        print(f'{col} is approximately symmetric with a skewness of {skewness:.2f}')
print (f'Skewness flagged columns: {flagged_skewness_train}')
print (f'Skewness flagged columns count: {len(flagged_skewness_train)}')


for df in [x_train_continuous, x_valid_continuous,x_test_train_continuous, x_test_continuous]:
    df[flagged_skewness_train] = np.log1p(df[flagged_skewness_train])
    
flagged_skewness_train_after_transforming = []

print (f'Skewness flagged after transforming')
for col in x_train_continuous.columns:
    skewness = x_train_continuous[col].skew().round(2)
    if skewness > 1 or skewness <= -1:
        print(f'{col} is highly skewed with a skewness of {skewness:.2f}')
        flagged_skewness_train_after_transforming.append(col)
    elif skewness > 0.5 or skewness < -0.5:
        print(f'{col} is moderately skewed with a skewness of {skewness:.2f}')
    else:
        print(f'{col} is approximately symmetric with a skewness of {skewness:.2f}')
print (f'Skewness flagged columns: {flagged_skewness_train_after_transforming}')
print (f'Skewness flagged columns count: {len(flagged_skewness_train_after_transforming)}')
print(f'flagged_skewness_train_after_transforming that has zero effect {set(flagged_skewness_train_after_transforming).intersection(zero_driven_features.columns)}')


#flagging skewness aside to the zero effect

flagged_skewness_train_zero_effect = []
targeted_zero_driven_features = x_train_continuous[zero_driven_features[flagged_skewness_train_after_transforming].columns]
targeted_zero_driven_features_final = targeted_zero_driven_features[targeted_zero_driven_features!=0]
for col in targeted_zero_driven_features.columns:
    skewness = targeted_zero_driven_features_final[col].skew().round(2)
    
    if skewness >= 1 or skewness <= -1:
        print(f'{col} is highly skewed with a skewness of {skewness:.2f}')
        flagged_skewness_train_zero_effect.append(col)
    elif skewness > 0.5 or skewness < -0.5:
        print(f'{col} is moderately skewed with a skewness of {skewness:.2f}')
    else:
        print(f'{col} is approximately symmetric with a skewness of {skewness:.2f}')
print (f'Skewness flagged columns: {flagged_skewness_train_zero_effect}')
print (f'Skewness flagged columns count: {len(flagged_skewness_train_zero_effect)}')


###detecting auto correlation




   







# ###

# # gonna preserve the zero driven features skewness tendency (we've already taken a log and they're still skewed)
# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
# countinuos featuer scaling 
from sklearn.preprocessing import  RobustScaler
scaler = RobustScaler()
x_train_continuous_scaled = scaler.fit_transform(x_train_continuous)
x_train_continuous_scaled = pd.DataFrame(x_train_continuous_scaled, columns=x_train_continuous.columns , index=x_train_continuous.index)
x_valid_continuous_scaled = scaler.transform(x_valid_continuous)
x_valid_continuous_scaled = pd.DataFrame(x_valid_continuous_scaled, columns=x_valid_continuous.columns , index=x_valid_continuous.index)
x_test_train_continuous_scaled = scaler.transform(x_test_train_continuous)
x_test_train_continuous_scaled = pd.DataFrame(x_test_train_continuous_scaled, columns=x_test_train_continuous.columns , index=x_test_train_continuous.index)
x_test_continuous_scaled = scaler.transform(x_test_continuous)
x_test_continuous_scaled = pd.DataFrame(x_test_continuous_scaled, columns=x_test_continuous.columns , index=x_test_continuous.index)
    
    
# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
#discrete features engineering

for col in x_train_discrete.columns:
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x_train_discrete[col], y_train)
    plt.title(f'Discrete Feature: {col}')
    print(f'Class count for {col}: {x_train_discrete[col].value_counts(normalize=True)*100}')
    # plt.show()
# for df in [x_train_discrete, x_valid_discrete, x_test_discrete]:
#     df = df.drop(columns=['PoolArea'])

for df in [x_train_discrete, x_valid_discrete , x_test_train_discrete, x_test_discrete]:
    for col in df.columns:
        if col in [year_features] :
            continue
        else:
            upper_limit = df[col].quantile(0.99)
            lower_limit = df[col].quantile(0.01)
            df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
            
non_year_features = x_train_discrete.columns.difference(year_features)
x_train_discrete_scaled = x_train_discrete.copy()
x_valid_discrete_scaled = x_valid_discrete.copy()
x_test_train_discrete_scaled = x_test_train_discrete.copy()
x_test_discrete_scaled = x_test_discrete.copy()
x_train_discrete_scaled[non_year_features] = scaler.fit_transform(x_train_discrete_scaled[non_year_features])
# x_train_discrete_scaled = pd.DataFrame(x_train_discrete_scaled, columns=x_train_discrete.columns , index=x_train_discrete.index)
x_valid_discrete_scaled[non_year_features] = scaler.transform(x_valid_discrete_scaled[non_year_features])
# x_valid_discrete_scaled = pd.DataFrame(x_valid_discrete_scaled, columns=x_valid_discrete.columns , index=x_valid_discrete.index)
x_test_train_discrete_scaled[non_year_features] = scaler.transform(x_test_train_discrete_scaled[non_year_features])
# x_test_train_discrete_scaled = pd.DataFrame(x_test_train_discrete_scaled, columns=x_test_train_discrete.columns , index=x_test_train_discrete.index)
x_test_discrete_scaled[non_year_features] = scaler.transform(x_test_discrete_scaled[non_year_features])
# x_test_discrete_scaled = pd.DataFrame(x_test_discrete_scaled, columns=x_test_discrete.columns , index=x_test_discrete.index)


# #####################################################################################################################################################################################################################################################################################################################################################################
# #####################################################################################################################################################################################################################################################################################################################################################################
# categorical features engineering


for col in x_train_categorical.columns:
    print(f'Class count for {col}: {x_train_categorical[col].value_counts(normalize=True)*100}')
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=col, y=y_train , data=x_train_categorical)
    plt.title(f'Violin Plots of {col}') 
    # plt.show()



def merge_rares_for_numerical_target (df_train : pd.DataFrame , df_valid : pd.DataFrame, df_test_train : pd.DataFrame ,  df_test : pd.DataFrame , date_features : list  , y_train : pd.Series , class_proportion_threshold : float = 0.05 , target_median_threshold : float = 0.05 , dominant_class_threshold : float = 0.95 ):
     dropped_columns = []
     merged_columns = []
     print(f'{df_train} initial columns {df_train.columns} and their count {len(df_train.columns)}')
     print(f'{df_valid} initial columns {df_valid.columns} and their count {len(df_valid.columns)}')
     print(f'{df_test_train} initial columns {df_test_train.columns} and their count {len(df_test_train.columns)}')
     print(f'{df_test} initial columns {df_test.columns} and their count {len(df_test.columns)}')
     
     if date_features is not None:
         df_train_without_date = df_train.drop(columns=date_features)
     else:
         df_train_without_date = df_train.copy()
        
     for col in df_train_without_date.columns:
        class_proportion = df_train[col].value_counts(normalize=True)
        rare_classes = class_proportion[class_proportion < class_proportion_threshold].index.tolist()
        non_rare_classes = class_proportion[class_proportion >= class_proportion_threshold].index.tolist()
        if non_rare_classes == [] or len(rare_classes) == 0 :
            # print(f'there are no rare classes in {col}')
            known_cats = set(df_train[col].dropna().unique())
            df_valid[col] = df_valid[col].mask(~df_valid[col].isin(known_cats), 'Other')
            df_test[col] = df_test[col].mask(~df_test[col].isin(known_cats), 'Other')
            df_test_train[col] = df_test_train[col].mask(~df_test_train[col].isin(known_cats), 'Other')
            continue
        highest_non_rare_class = max(non_rare_classes, key=lambda x: class_proportion[x])
        highest_non_rare_proportion = class_proportion[highest_non_rare_class]
        if  highest_non_rare_proportion >= dominant_class_threshold or (len(non_rare_classes) == 1  and len(class_proportion) <= 2) :
            # print(f'{col} has a dominant class {highest_non_rare_class} and it is going to be dropped')
            dropped_columns.append(col)
            df_train = df_train.drop(columns=[col])
            df_valid = df_valid.drop(columns=[col])
            df_test = df_test.drop(columns=[col])
            df_test_train = df_test_train.drop(columns=[col])
            continue
            
        else:
            smallest_non_rare_class = min(non_rare_classes, key=lambda x: class_proportion[x])
            rare_target_median = y_train[df_train[col].isin(rare_classes)].median()
            non_rare_target_median = y_train[df_train[col] == smallest_non_rare_class].median()
            scaled_diff = abs(rare_target_median - non_rare_target_median) / non_rare_target_median
            
            if len(rare_classes) > 1:
                        # print(f'there are {len(rare_classes)} rare classes in {col} and they are {rare_classes} and they are going to be merged')
                        merged_columns.append(col)
                       
                        df_train[col] = df_train[col].mask(df_train[col].isin(rare_classes), 'Other')
                        known_cats = set(df_train[col].dropna().unique())
                        
                        
                        df_valid[col] = df_valid[col].mask(~df_valid[col].isin(known_cats), 'Other') #merge any unseen in the valid set category into other
                        
                        df_test_train[col] = df_test_train[col].mask(~df_test_train[col].isin(known_cats), 'Other') 
                       
                        df_test[col] = df_test[col].mask(~df_test[col].isin(known_cats), 'Other') #merge any unseen category in the test set into other
                    
            elif  len(rare_classes) == 1 and len(class_proportion) > 2 and scaled_diff <= target_median_threshold:
                        merged_columns.append(col)
                        # print(f"there's only one rare class in {col} and it is {rare_classes} and it is going to be merged")
                        df_train[col] = df_train[col].mask(df_train[col].isin(rare_classes), 'Other')
                        known_cats = set(df_train[col].dropna().unique())
                      
                        df_valid[col] = df_valid[col].mask(~df_valid[col].isin(known_cats), 'Other')
                        df_test_train[col] = df_test_train[col].mask(~df_test_train[col].isin(known_cats), 'Other')
                        df_test[col] = df_test[col].mask(~df_test[col].isin(known_cats), 'Other')
            elif len(rare_classes) == 1 and len(class_proportion) > 2 and scaled_diff > target_median_threshold:
                        # print(f"there's only one rare class in {col} and it is {rare_classes} and it is not going to be merged")
                        known_cats = set(df_train[col].dropna().unique())
                        df_valid[col] = df_valid[col].mask(~df_valid[col].isin(known_cats), 'Other')
                        df_test[col] = df_test[col].mask(~df_test[col].isin(known_cats), 'Other')
                        df_test_train[col] = df_test_train[col].mask(~df_test_train[col].isin(known_cats), 'Other')
                        continue 
       
      
        
                
       
     print(f'dropped_columns {dropped_columns} and their count {len(dropped_columns)}')
     print(f'merged_columns {merged_columns} and their count {len(merged_columns)}')
     print(f'{df_train}  finalcolumns {df_train.columns} and their count {len(df_train.columns)}')
     print(f'{df_valid} final columns {df_valid.columns} and their count {len(df_valid.columns)}')
     print(f'{df_test_train} final columns {df_test_train.columns} and their count {len(df_test_train.columns)}')
     print(f'{df_test} final columns {df_test.columns} and their count {len(df_test.columns)}')
     return df_train, df_valid, df_test_train ,  df_test , dropped_columns , merged_columns


# merge_rares_for_numerical_target(df_train = x_train_categorical,df_valid = x_valid_categorical, df_test = x_test_categorical , df_test_train = x_test_train_categorical, date_features = ['DateSold'] , y_train = y_train , class_proportion_threshold= 0.05 , target_median_threshold = 0.1 , dominant_class_threshold = 0.9)
dropped_cat_columns = []
merged_cat_columns = []
x_train_categorical , x_valid_categorical ,x_test_train_categorical ,  x_test_categorical , dropped_cat_columns , merged_cat_columns = merge_rares_for_numerical_target(df_train = x_train_categorical  ,df_valid = x_valid_categorical,df_test_train = x_test_train_categorical ,  df_test = x_test_categorical, date_features = ['DateSold' , 'MoSold'] , y_train = y_train , class_proportion_threshold= 0.05 , target_median_threshold = 0.1 , dominant_class_threshold = 0.9)

    
# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
# #final data representation

print(f'y_train skewness {y_train.skew()}')
y_train_log = np.log1p(y_train)
y_valid_log = np.log1p(y_valid)
y_test_train_log = np.log1p(y_test_train)

print(f'y_train_log skewness {y_train_log.skew()}')
x_train_final = pd.concat([x_train_categorical, x_train_continuous_scaled , x_train_discrete_scaled], axis=1)
x_valid_final = pd.concat([x_valid_categorical, x_valid_continuous_scaled , x_valid_discrete_scaled], axis=1)
x_test_train_final = pd.concat([x_test_train_categorical, x_test_train_continuous_scaled , x_test_train_discrete_scaled], axis=1)
x_test_final = pd.concat([x_test_categorical, x_test_continuous_scaled , x_test_discrete_scaled], axis=1)


print(f'train set final shape {x_train_final.shape}')
print(f'valid set final shape {x_valid_final.shape}')
print(f'test set final shape {x_test_final.shape}')
print(f'train set final columns {x_train_final.columns}')
print(f'valid set final columns {x_valid_final.columns}')
print(f'test set final columns {x_test_final.columns}')
print(f'are my 4 sets the same {x_train_final.columns.equals(x_valid_final.columns) and x_train_final.columns.equals(x_test_final.columns) and x_train_final.columns.equals(x_test_train_final.columns)}')
print (f'train set final missings {x_train_final.isna().sum()}')
print (f'valid set final missings {x_valid_final.isna().sum()}')
print (f'test set final missings {x_test_final.isna().sum()}')
print(f'train set final info {x_train_final.info()}')
print(f'valid set final info {x_valid_final.info()}')
print(f'test set final info {x_test_final.info()}')







# ################################################################################################################################################################################################################################################################
# ################################################################################################################################################################################################################################################################
# #feature selection
from sklearn.feature_selection import SequentialFeatureSelector , RFECV , RFE  
from sklearn.linear_model import LassoCV , ElasticNet , LinearRegression

x_train_numeric = x_train_final.select_dtypes(include=['int64', 'float64'])
x_train_categorical = x_train_final.select_dtypes(exclude=['int64', 'float64'])
full_df_numeric = pd.concat([x_train_numeric, y_train_log], axis=1)
corr_matrix = full_df_numeric.corr(method='pearson' , numeric_only= True )
corr_heatmap=sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix', fontsize=16) 
# plt.show()



from sklearn.linear_model import Lasso , SGDRegressor , Ridge , ElasticNetCV , LassoCV , RidgeCV
import statsmodels.api as sm
x_train_encoded = x_train_final.copy()
x_train_encoded = x_train_encoded.drop(columns=['DateSold'])
x_valid_encoded = x_valid_final.copy()
x_valid_encoded = x_valid_encoded.drop(columns=['DateSold'])
x_test_train_encoded = x_test_train_final.copy()
x_test_train_encoded = x_test_train_encoded.drop(columns=['DateSold'])
x_test_encoded = x_test_final.copy()
x_test_encoded = x_test_encoded.drop(columns=['DateSold'])

x_train_encoded = pd.get_dummies(x_train_encoded , drop_first=True , dtype=int)
x_valid_encoded = pd.get_dummies(x_valid_encoded , drop_first=True , dtype=int)
x_test_train_encoded = pd.get_dummies(x_test_train_encoded , drop_first=True , dtype=int)
x_test_encoded = pd.get_dummies(x_test_encoded , drop_first=True , dtype=int)
x_train_numeric_encoded = x_train_encoded[x_train_numeric.columns]
x_train_categorical_encoded = x_train_encoded.drop(columns=x_train_numeric.columns)
x_valid_numeric_encoded = x_valid_encoded[x_valid_numeric.columns]
x_valid_categorical_encoded = x_valid_encoded.drop(columns=x_valid_numeric.columns)
x_test_train_numeric_encoded = x_test_train_encoded[x_test_train_numeric.columns]
x_test_train_categorical_encoded = x_test_train_encoded.drop(columns=x_test_train_numeric.columns)
x_test_numeric_encoded = x_test_encoded[x_test_numeric.columns]
x_test_categorical_encoded = x_test_encoded.drop(columns=x_test_numeric.columns)




alphas = np.logspace(-5, 1, 50)


Lasso_model = RidgeCV(alphas = alphas  ,  scoring = 'neg_mean_squared_error' )# elasticnet penalty because of the moderate-high multicollinearity between features
RFECV_model  = RFECV(estimator =Lasso_model,   cv=3, scoring='neg_mean_squared_error' , n_jobs=-1, step = 5)
RFECV_model.fit(x_train_encoded, y_train_log)
RFECV_selected_features = x_train_encoded.columns[RFECV_model.get_support()]
coefs = RFECV_model.estimator_.coef_.ravel()
important_features = RFECV_selected_features[coefs != 0]
best_alpha = RFECV_model.estimator_.alpha_
features_importance = pd.DataFrame({
    'Feature': important_features,
    'Importance': coefs[coefs != 0],
    'Best alpha': best_alpha
}).sort_values(by='Importance', ascending=False)
print(f'Selected features using RFECV : {important_features.tolist()} and their count is {len(important_features)} in which {len(important_features.intersection(x_train_numeric_encoded.columns))} are numerical and {len(important_features.difference(x_train_numeric_encoded.columns))} are categorical')
print(f'RFECV feature importance: {features_importance}')
print(' and their R2 score is ' ,{RFECV_model.score(x_train_encoded, y_train_log)})






# # ########################################################################################################################################################################################### 
#simple linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score , mean_squared_error , root_mean_squared_error
from statsmodels.api import  OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler

x_train_selected_features = x_train_encoded[important_features]
x_valid_selected_features = x_valid_encoded[important_features]
x_test_train_selected_features = x_test_train_encoded[important_features]
x_test_selected_features = x_test_encoded[important_features]
#sklearn
linear_regression = LinearRegression(n_jobs=-1)

ploymodel = PolynomialFeatures(degree=2 , include_bias=False)
continuous_features = x_train_selected_features.columns.intersection(x_train_continuous.columns)
x_train_cont = x_train_selected_features[continuous_features].values
x_train_cat = x_train_selected_features.drop(columns=continuous_features)
x_valid_cont = x_valid_selected_features[continuous_features].values
x_valid_cat = x_valid_selected_features.drop(columns=continuous_features)
x_test_train_cont = x_test_train_selected_features[continuous_features].values
x_test_train_cat = x_test_train_selected_features.drop(columns=continuous_features)
x_test_cont = x_test_selected_features[continuous_features].values
x_test_cat = x_test_selected_features.drop(columns=continuous_features)
x_train_poly = ploymodel.fit_transform(x_train_cont)
x_valid_poly = ploymodel.transform(x_valid_cont)
x_test_train_poly = ploymodel.transform(x_test_train_cont)
x_test_poly = ploymodel.transform(x_test_cont)
x_train_poly_scaled = StandardScaler().fit_transform(x_train_poly)
x_valid_poly_scaled = StandardScaler().fit_transform(x_valid_poly)
x_test_train_poly_scaled = StandardScaler().fit_transform(x_test_train_poly)
x_test_poly_scaled = StandardScaler().fit_transform(x_test_poly)
x_train_all_selected = np.hstack((x_train_cat, x_train_poly_scaled))
x_valid_all_selected = np.hstack((x_valid_cat, x_valid_poly_scaled))
x_test_train_all_selected = np.hstack((x_test_train_cat, x_test_train_poly_scaled))
x_test_all_selected = np.hstack((x_test_cat, x_test_poly_scaled))


linear_regression.fit(x_train_selected_features, y_train_log)
y_train_pred = linear_regression.predict(x_train_selected_features)

train_residuals = y_train_log - y_train_pred
##checking assumptions
plt.close('all')
plt.figure(figsize=(7, 4))
sns.scatterplot(x=y_train_pred, y=train_residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Linearity & homoscedasticity (train)')
plt.show()
dw_score = durbin_watson(train_residuals)
print(f"Durbin-Watson Statistic: {dw_score:.3f}")


sm.qqplot(y_train_log, line='s', ax=plt.gca())
plt.title('Normal Q-Q Plot (train)')
plt.show()



print(f'Linear regression coefficients are {linear_regression.coef_} and the intercept is {linear_regression.intercept_}')
print(f'R2 score of linear regression is {linear_regression.score(x_train_selected_features, y_train_log)}')
print(f'MSE of linear regression is {root_mean_squared_error(y_train_log, y_train_pred) }')
y_actual_array = np.array(y_train_log)
sorted_index = np.argsort(y_train_pred)
y_actual_sorted = y_actual_array[sorted_index]
y_pred_train_sorted = y_train_pred[sorted_index]
sample_indices = range(len(y_actual_sorted))
plt.close('all')
plt.figure(figsize=(10, 6))
plt.scatter(sample_indices, y_actual_sorted, color='blue', label='Actual')
plt.plot(sample_indices, y_pred_train_sorted, color='red', label='Predicted')
plt.title('Linear Regression train set')
plt.xlabel('Sample Index')
plt.ylabel('Sale Price (log)')
plt.show()


#validation set


y_valid_pred = linear_regression.predict(x_valid_selected_features)
valid_residuals = y_valid_log - y_valid_pred
##checking assumptions
plt.figure(figsize=(7, 4))
sns.scatterplot(x=y_valid_pred, y=valid_residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Linearity & homoscedasticity (validation)')
plt.show()
dw_score = durbin_watson(valid_residuals)
print(f"Durbin-Watson Statistic: {dw_score:.3f}")
sm.qqplot(y_valid_log, line='s', ax=plt.gca())
plt.title('Normal Q-Q Plot (validation)')
plt.show()

print(f'linear regression validation set MSE is {root_mean_squared_error(y_valid_log, y_valid_pred)}')
y_valid_array = np.array(y_valid_log)
sorted_index_valid = np.argsort(y_valid_pred)
y_valid_sorted = y_valid_array[sorted_index_valid]
y_valid_pred_sorted = y_valid_pred[sorted_index_valid]
sample_indices_valid = range(len(y_valid_sorted))

plt.figure(figsize=(10, 6))
plt.scatter(sample_indices_valid, y_valid_sorted, color='blue', label='Actual')
plt.plot(sample_indices_valid, y_valid_pred_sorted, color='red', label='Predicted')
plt.title('Linear Regression validation set')
plt.xlabel('Sample Index')
plt.ylabel('Sale Price (log)')
plt.show()

#train_test set
y_train_test_pred_log = linear_regression.predict(x_test_train_selected_features)
train_test_residuals = y_test_train_log - y_train_test_pred_log
##checking assumptions
plt.figure(figsize=(7, 4))
sns.scatterplot(x=y_train_test_pred_log, y=train_test_residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Linearity & homoscedasticity (train_test)')
plt.show()
dw_score = durbin_watson(train_test_residuals)
print(f"Durbin-Watson Statistic: {dw_score:.3f}")
sm.qqplot(y_test_train_log, line='s', ax=plt.gca())
plt.title('Normal Q-Q Plot (train_test)')
plt.show()

print(f'linear regression train_test set MSE is {root_mean_squared_error(y_test_train_log, y_train_test_pred_log)}')
y_train_test_array = np.array(y_test_train_log)
sorted_index_train_test = np.argsort(y_train_test_pred_log)
y_train_test_sorted = y_train_test_array[sorted_index_train_test]
y_train_test_pred_sorted = y_train_test_pred_log[sorted_index_train_test]
sample_indices_train_test = range(len(y_train_test_sorted))

plt.figure(figsize=(10, 6))
plt.scatter(sample_indices_train_test, y_train_test_sorted, color='blue', label='Actual')
plt.plot(sample_indices_train_test, y_train_test_pred_sorted, color='red', label='Predicted')
plt.title('Linear Regression train_test set')
plt.xlabel('Sample Index')
plt.ylabel('Sale Price (log)')  
plt.show()

















y_test_pred_log = linear_regression.predict(x_test_selected_features)
y_test_pred = np.expm1(y_test_pred_log)
submission_linear_reg = pd.DataFrame({'Id': test_id, 'SalePrice': y_test_pred})
submission_linear_reg.to_csv('submission_linear_reg.csv', index=False)
submission_linear_reg.to_csv('C:\\Users\\user\\Downloads\\submission_linear_reg_refined.csv', index=False)
print(f'linear regression test predictions for test set  are {submission_linear_reg.head()}')


