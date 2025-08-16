# Multible linear regression with sklearn 

# some times you'll need to preform train test split to handle the over and under fitting problem

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt  
import seaborn as sb
sb.set()

from sklearn.linear_model import LinearRegression               # <- to applly the linear regression
from sklearn.metrics import r2_score                            # <- to get the r2 value
from sklearn.model_selection import train_test_split            # <- train_test_split
from sklearn.preprocessing import StandardScaler                # <- apply standardization 
from statsmodels.stats.outliers_influence import variance_inflation_factor # <- Multicolinearity

def show(data_no_mv, col, target):
    sb.scatterplot(x=data_no_mv[col], y=data_no_mv[target])
    plt.title(f"{col} vs {target}")
    plt.xlabel(col)
    plt.ylabel(target)
    plt.show()

def remove_out_layers(data, features):
    for i in features:
        if i in data.columns and is_numeric_dtype(data[i]):
            skew = data[i].skew()

            if -0.5 <= skew <= 0.5:
                print(f'No skew for the {i} [{skew}]')
                continue  # symmetric, no outlier removal

            elif 0.5 < skew < 1:
                q = data[i].quantile(0.99)
                data = data[data[i] < q]

            elif skew >= 1:
                q = data[i].quantile(0.98)
                data = data[data[i] < q]

            elif -1 < skew < -0.5:
                q = data[i].quantile(0.01)
                data = data[data[i] > q]

            elif skew <= -1:
                q = data[i].quantile(0.02)
                data = data[data[i] > q]
            print(f'skew handled for {i} [{skew}]')

    return data

def MultibleLinearRegression(fileName, target, features):
    # 🔹 Load training data
    data = pd.read_csv(fileName)

    # get not usefull columns
    print(data.describe(include='all'))
    columns_to_drop = [f.strip() for f in input('what is the columns U C it''s not usfull logicaly (use "-" between names): ').split('-')]
    
    # drop mot usefull columns
    if columns_to_drop:
        data.drop(columns=columns_to_drop, inplace=True)
        features = [f for f in features if f not in columns_to_drop]
        print(f'columns droped successfuly')
    else:
        data = data.copy()

    # handle missing values
    if data.isnull().values.any():
        data.dropna(axis=0, inplace=True)
        print(f'missing values handled successfully')

    # remove outlayers
    features = features + [target]
    data = remove_out_layers(data, features)
    features.remove(target)
    
    # handle linearity
    target_was_logged = False  # Flag to track if target got log-transformed

    for i in features:
        if is_numeric_dtype(data[str(i)]):
            while True:
                    show(data, i, target)
                    x = int(input('📈 Is this linear? [1: Yes, 2: No]: '))
                    
                    if x == 1:
                        break
                    
                    elif x == 2:
                        if not target_was_logged:
                            # Apply log on target
                            target_log = target + '_log'
                            data[target_log] = np.log(data[target])
                            data.drop(columns=target, inplace=True)
                            target = target_log
                            target_was_logged = True
                            print(f'✅ Log transformation applied on target: {target}')
                        else:
                            # Apply log on feature
                            feature_log = i + '_log'
                            data[feature_log] = np.log(data[i])
                            data.drop(columns=i, inplace=True)
                            features[features.index(i)] = feature_log
                            print(f'✅ Log transformation applied on feature: {i}')
                            break

    
    # Milticolinearity handling
    while True:
        X = data[[x for x in features if is_numeric_dtype(data[x])]]
        vif = pd.DataFrame()
        vif["Features"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        max_vif = vif["VIF"].max()
        if max_vif > 10:
            drop_feature = vif.sort_values("VIF", ascending=False).iloc[0]["Features"]
            data.drop(columns=drop_feature, inplace=True)
            features.remove(drop_feature)
            print(f"Dropped {drop_feature} due to high VIF: {max_vif:.2f}")
        else:
            break


    # get dummies
    data = pd.get_dummies(data, drop_first=False, dtype=int)
    features = [col for col in data.columns if col != target_log]

    x = data[features]
    y = data[target]

    # 🔹 Apply standardization (scaling)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    print(data.head())

    # 🔹 Split into training and test sets (80% train, 20% test)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # 🔹 Train the model on training data
    reg = LinearRegression()
    reg.fit(x_train, y_train)

    # 🔹 Predict on test data
    y_pred = reg.predict(x_test)

    # 🔹 R² score on test set
    r2 = r2_score(y_test, y_pred)
    print(f"🧪 Test R² score: {r2:.2f}")

    # 🔹 Optional: Show coefficients
    '''print("\n📈 Model Coefficients:")
    for i, col in enumerate(features):
        print(f"{col}: {reg.coef_[i]:.4f}")
    print(f"Intercept: {reg.intercept_:.4f}")
    '''
    # 🔹 Regression summary table
    reg_summary = pd.DataFrame(['Bias'] + features, columns=['Feature'])
    reg_summary['Weight (coef)'] = [reg.intercept_] + list(reg.coef_)
    print("\n📋 Regression Summary:")
    print(reg_summary)

    # 🔹 Predict on new data
    i = input("\n📂 Enter the file name to predict on: ")
    new_data = pd.read_csv(i)
    new_x_scaled = scaler.transform(new_data[features])
    predictions = reg.predict(new_x_scaled)

        # If the target was transformed into log, convert predictions back
    if 'log' in target:
        original_target = target.replace('_log', '')
        new_data['predicted_' + original_target] = np.exp(predictions)
    else:
        new_data['predicted_' + target] = predictions

    print("\n📊 Predictions:")
    print(new_data)

    output_name = input("\n💾 Enter file name to save predictions (without .csv): ")
    new_data.to_csv(output_name + ".csv", index=False)
    print(f"\n✅ Predictions saved to {output_name}.csv")


def get_info():
    x1 = input('📁 Enter the sample data file name: ')
    x2 = input('🎯 Enter the target variable name: ')
    x3 = [f.strip() for f in input('🔢 Enter the feature vars (use "-" between names): ').split('-')]
    MultibleLinearRegression(x1, x2, x3)

get_info()
