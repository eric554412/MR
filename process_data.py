import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import numpy as np

pd.set_option('display.max_row', None)
pd.set_option('display.max_columns', None)

def merge_data(data, factor):
    '''
    將數據與因子資料結合
    '''
    df = data.copy()
    df = pd.merge(data, factor, on='年月')
    df.drop(columns={'證券代碼_y'}, inplace=True)
    df.rename(columns={'證券代碼_x': '證券代碼'}, inplace=True)
    return df


def descriptive_statistics(data):
    '''
    計算敘述性統計
    '''
    return_mean = data.groupby('證券代碼')['報酬率％_月'].mean()
    return_var = data.groupby('證券代碼')['報酬率％_月'].var()
    descriptive_df = pd.merge(return_mean, return_var, on='證券代碼')
    descriptive_df.rename(
        columns={'報酬率％_月_x': '平均報酬', '報酬率％_月_y': '報酬變異數'}, inplace=True)
    cov_martic = df.pivot(index='年月', columns='證券代碼', values='報酬率％_月').cov()
    return descriptive_df, cov_martic


def drop_company(data):
    '''
    drop掉缺失值過多的公司(大陸公司)還有因子
    '''
    companies_withnan = data[data.isna().any(axis=1)]['證券代碼'].unique()
    data = data[~data['證券代碼'].isin(companies_withnan)]
    company_count = data['證券代碼'].value_counts()
    valid_company = company_count[company_count == 132].index
    data = data[data['證券代碼'].isin(valid_company)]
    data.drop(columns=['本益比-TEJ'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def split_data(data, return_time, predictor_time):
    train_data_return = data.loc[data['年月'] ==
                                 return_time, ['證券代碼', '年月', '報酬率％_月']]
    train_data_perdictor = data.loc[data['年月'] == predictor_time].drop(
        columns=['收盤價(元)_月', '報酬率％_月', '年月'])
    regression_data = pd.merge(
        train_data_return, train_data_perdictor, on='證券代碼')
    x = regression_data.drop(columns=['報酬率％_月', '證券代碼', '年月'])
    y = regression_data['報酬率％_月']
    return x, y


def run_regression(x_train, y_train):
    model = sm.OLS(y_train, x_train).fit()
    # 預測值
    y_train_pred = model.predict(x_train)

    # 計算 RMSE
    rmse_train = np.sqrt(np.sum((y_train - y_train_pred) ** 2))

    # 計算 leverage scores(loo)
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag

    # 計算 Leave-One-Out誤差
    residuals = y_train_pred - y_train
    loo_error = (leverage / (1 - leverage)) * residuals

    return model, leverage, loo_error, rmse_train


def test_estimated(x_test, y_test):
    '''
    測試資料
    '''
    model, leverage, loo_error, rmse_train = run_regression(x_train, y_train)
    x_test = sm.add_constant(x_test)
    y_test_pred = model.predict(x_test)

    rmse_test = np.sqrt(np.sum((y_test - y_test_pred) ** 2))
    return rmse_test


def run_regression_with_filter(x_train, y_train, x_test, y_test):
    '''
    filtered data and repeat regression
    '''
    model, leverage, loo_error, rmse_train = run_regression(x_train, y_train)

    x_train_filtered = x_train.copy()
    x_train_filtered['leverage'] = leverage

    p = x_train_filtered.shape[1] - 1  # 扣掉 leverage
    n = y_train.shape[0]

    # 過濾條件
    filter_condition = x_train_filtered['leverage'] <= 2 * p / n

    x_train_filtered = x_train_filtered[filter_condition].drop(columns=[
                                                               'leverage'])
    y_train_filtered = y_train[filter_condition]

    model2 = sm.OLS(y_train_filtered, x_train_filtered).fit()

    y_train_filtered_pred = model2.predict(x_train_filtered)

    # 計算新的rmse, leverage, loo error
    filtered_train_rmse = np.sqrt(
        np.sum((y_train_filtered - y_train_filtered_pred) ** 2))
    influence = model2.get_influence()
    leverage_filtered = influence.hat_matrix_diag
    residuals_filtered = y_train_filtered_pred - y_train_filtered
    loo_error_filtered = (leverage_filtered /
                          (1 - leverage_filtered)) * residuals_filtered

    # 計算新的test_rmse
    y_test_filtered_pred = model2.predict(x_test)
    filtered_test_rmse = np.sqrt(np.sum((y_test - y_test_filtered_pred) ** 2))

    return filtered_train_rmse, filtered_test_rmse, leverage_filtered, loo_error_filtered


def run_regression_with_filterloo(x_train, y_train, x_test, y_test):
    model, leverage, loo_error, rmse_train = run_regression(x_train, y_train)
    x_train['loo_error'] = loo_error

    # drop 絕對值前三大的loo_error 觀測值
    largest_3_indices = loo_error.abs().nlargest(3).index

    x_train_filteredloo = x_train.drop(
        index=largest_3_indices).drop(columns=['loo_error'])
    y_train_filteredloo = y_train.drop(index=largest_3_indices)

    model_filtered = sm.OLS(y_train_filteredloo, x_train_filteredloo).fit()

    y_train_filteredloo_pred = model_filtered.predict(x_train_filteredloo)

    # 計算新的rmse, leverage, loo error
    filteredloo_train_rmse = np.sqrt(
        np.sum((y_train_filteredloo - y_train_filteredloo_pred) ** 2))
    influenceloo = model_filtered.get_influence()
    leverage_filteredloo = influenceloo.hat_matrix_diag
    residuals_filteredloo = y_train_filteredloo_pred - y_train_filteredloo
    loo_error_filteredloo = (leverage_filteredloo /
                             (1 - leverage_filteredloo)) * residuals_filteredloo

    # 計算新的test_rmse
    y_test_filtered_pred = model_filtered.predict(x_test)
    filteredloo_test_rmse = np.sqrt(
        np.sum((y_test - y_test_filtered_pred) ** 2))

    return filteredloo_train_rmse, filteredloo_test_rmse, leverage_filteredloo, loo_error_filteredloo




if __name__ == '__main__':
    data = pd.read_csv('/Users/huyiming/Downloads/20250305084429_close.csv',
                       encoding='utf-16', sep='\t')

    factor = pd.read_csv('/Users/huyiming/Downloads/20250305064046_factor.csv',
                         encoding='utf-16', sep='\t')

    df = merge_data(data, factor)
    df = drop_company(df)

    x_train, y_train = split_data(df, 202412, 202411)
    model, leverage, loo_error, rmse_train = run_regression(x_train, y_train)
    print(f'RMSE: {rmse_train:.4f}')
    print(f'leverage scores分佈: {np.percentile(leverage, [1, 20, 50, 75, 99])}')
    print(f'leverage max: {np.max(leverage)}')
    print(f'loo error分佈: {np.percentile(loo_error, [1, 20, 50, 75, 99])}')
    print(f'loo error max: {np.max(loo_error)}')
    print('*********************')

    x_test, y_test = split_data(df, 202501, 202412)
    rmse_test = test_estimated(x_test, y_test)
    print(f'Out of sample Rmse: {rmse_test}')
    print('*********************')

    filtered_train_rmse, filtered_test_rmse, leverage_filtered, loo_error_filtered = run_regression_with_filter(
        x_train, y_train, x_test, y_test)
    print(f'after filter train RMSE: {filtered_train_rmse}')
    print(
        f'after filter leverage scores分佈: {np.percentile(leverage_filtered, [1, 20, 50, 75, 99])}')
    print(f'after filter leverage max: {np.max(leverage_filtered)}')
    print(
        f'after filter loo error分佈: {np.percentile(loo_error_filtered, [1, 20, 50, 75, 99])}')
    print(f'after filter loo error max: {np.max(loo_error_filtered)}')
    print(f'after filter test RMSE: {filtered_test_rmse}')
    print('*********************')

    filteredloo_train_rmse, filteredloo_test_rmse, leverage_filteredloo, loo_error_filteredloo = run_regression_with_filterloo(
        x_train, y_train, x_test, y_test)
    print(f'after filteloo train RMSE: {filteredloo_train_rmse}')
    print(
        f'after filterloo leverage scores分佈: {np.percentile(leverage_filteredloo, [1, 20, 50, 75, 99])}')
    print(f'after filterloo leverage max: {np.max(leverage_filteredloo)}')
    print(
        f'after filterloo loo error分佈: {np.percentile(loo_error_filteredloo, [1, 20, 50, 75, 99])}')
    print(f'after filterloo loo error max: {np.max(loo_error_filteredloo)}')
    print(f'after filterloo test RMSE: {filteredloo_test_rmse}')
    
    data_dict = {
        "指標": [
            "Train RMSE", "Test RMSE",
            "Leverage Score 分佈", "Leverage Score Max",
            "LOO Error 分佈", "LOO Error Max"
        ],
        "原始模型": [
            163.4048, 128.2772,
            [0.0036, 0.0053, 0.0080, 0.0141, 0.2257], 0.2963,
            [-0.8887, -0.0335, 0.0047, 0.0375, 0.9338], 3.3045
        ],
        "過濾 Leverage": [
            160.8392, 127.4231,
            [0.0038, 0.0063, 0.0101, 0.0184, 0.1307], 0.1587,
            [-1.0138, -0.0382, 0.0072, 0.0491, 0.6852], 2.4848
        ],
        "過濾 LOO Error": [
            161.8945, 131.0010,
            [0.0036, 0.0056, 0.0086, 0.0146, 0.2038], 0.2699,
            [-0.9888, -0.0337, 0.0055, 0.0397, 0.5950], 2.5328
        ]
    }
    
    df = pd.DataFrame(data_dict)
    print(df)