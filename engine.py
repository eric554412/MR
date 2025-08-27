
import pandas as pd
import numpy as np
import statsmodels.api as sm

class AnalysisEngine:
    """
    一個封裝了資料處理與迴歸分析流程的引擎。
    """

    def __init__(self, data_path, factor_path):
        """
        初始化引擎。

        :param data_path: 收盤價資料 CSV 的路徑。
        :param factor_path: 因子資料 CSV 的路徑。
        """
        self.data_path = data_path
        self.factor_path = factor_path
        self.df = None
        self.results = {
            "指標": [
                "Train RMSE", "Test RMSE",
                "Leverage Score 分佈 (百分位)", "Leverage Score Max",
                "LOO Error 分佈 (百分位)", "LOO Error Max"
            ]
        }

    def _load_and_merge_data(self):
        """
        載入並合併資料。
        """
        try:
            data = pd.read_csv(self.data_path, encoding='utf-16', sep='\t')
            factor = pd.read_csv(self.factor_path, encoding='utf-16', sep='\t')
        except FileNotFoundError as e:
            raise Exception(f"錯誤：找不到檔案 - {e}")
        except Exception as e:
            raise Exception(f"讀取檔案時發生錯誤: {e}")

        df = pd.merge(data, factor, on='年月')
        if '證券代碼_y' in df.columns:
            df.drop(columns={'證券代碼_y'}, inplace=True)
        if '證券代碼_x' in df.columns:
            df.rename(columns={'證券代碼_x': '證券代碼'}, inplace=True)
        return df

    def _preprocess_data(self, df):
        """
        資料預處理，移除不需要的公司與欄位。
        """
        # drop掉缺失值過多的公司
        companies_withnan = df[df.isna().any(axis=1)]['證券代碼'].unique()
        df = df[~df['證券代碼'].isin(companies_withnan)]
        
        # 篩選掉資料月份不完整的公司
        company_month_counts = df['證券代碼'].value_counts()
        if not company_month_counts.empty:
            max_months = company_month_counts.iloc[0]
            valid_companies = company_month_counts[company_month_counts == max_months].index
            df = df[df['證券代碼'].isin(valid_companies)]

        if '本益比-TEJ' in df.columns:
            df.drop(columns=['本益比-TEJ'], inplace=True)
        
        df.reset_index(drop=True, inplace=True)
        self.df = df
        return self.df

    def _split_data(self, return_time, predictor_time):
        """
        根據時間切割訓練與測試資料。
        """
        train_data_return = self.df.loc[self.df['年月'] == return_time, ['證券代碼', '年月', '報酬率％_月']]
        train_data_predictor = self.df.loc[self.df['年月'] == predictor_time].drop(
            columns=['收盤價(元)_月', '報酬率％_月', '年月'])
        
        regression_data = pd.merge(train_data_return, train_data_predictor, on='證券代碼')
        
        x = regression_data.drop(columns=['報酬率％_月', '證券代碼', '年月'])
        y = regression_data['報酬率％_月']
        
        # 為模型加入常數項
        x = sm.add_constant(x)
        
        return x, y

    def _run_single_regression(self, x_train, y_train):
        """
        執行單次 OLS 迴歸並計算相關指標。
        """
        model = sm.OLS(y_train, x_train).fit()
        y_train_pred = model.predict(x_train)
        
        rmse_train = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        
        residuals = y_train - y_train_pred # 注意：殘差是 y_true - y_pred
        loo_error = residuals / (1 - leverage)

        return model, leverage, loo_error, rmse_train
    
    def _evaluate_on_test(self, model, x_test, y_test):
        """
        在測試集上評估模型。
        """
        y_test_pred = model.predict(x_test)
        rmse_test = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        return rmse_test

    def run_analysis(self, train_return_time=202412, train_predictor_time=202411, 
                     test_return_time=202501, test_predictor_time=202412):
        """
        執行完整的分析流程。
        """
        base_df = self._load_and_merge_data()
        self.df = self._preprocess_data(base_df)

        if self.df.empty:
            raise ValueError("資料預處理後為空，請檢查輸入檔案的內容與格式。")

        # 準備資料
        x_train, y_train = self._split_data(train_return_time, train_predictor_time)
        x_test, y_test = self._split_data(test_return_time, test_predictor_time)

        # 1. 原始模型
        model, leverage, loo_error, rmse_train = self._run_single_regression(x_train, y_train)
        rmse_test = self._evaluate_on_test(model, x_test, y_test)
        self.results["原始模型"] = [
            f"{rmse_train:.4f}", f"{rmse_test:.4f}",
            f"{np.percentile(leverage, [25, 50, 75])}", f"{np.max(leverage):.4f}",
            f"{np.percentile(loo_error, [25, 50, 75])}", f"{np.max(np.abs(loo_error)):.4f}"
        ]

        # 2. 過濾 Leverage
        p = x_train.shape[1]
        n = y_train.shape[0]
        filter_condition = leverage <= 2 * p / n
        x_train_filtered = x_train[filter_condition]
        y_train_filtered = y_train[filter_condition]

        model2, _, _, rmse_train_filtered = self._run_single_regression(x_train_filtered, y_train_filtered)
        rmse_test_filtered = self._evaluate_on_test(model2, x_test, y_test)
        influence2 = model2.get_influence()
        leverage2 = influence2.hat_matrix_diag
        residuals2 = y_train_filtered - model2.predict(x_train_filtered)
        loo_error2 = residuals2 / (1 - leverage2)
        
        self.results["過濾 Leverage"] = [
            f"{rmse_train_filtered:.4f}", f"{rmse_test_filtered:.4f}",
            f"{np.percentile(leverage2, [25, 50, 75])}", f"{np.max(leverage2):.4f}",
            f"{np.percentile(loo_error2, [25, 50, 75])}", f"{np.max(np.abs(loo_error2)):.4f}"
        ]

        # 3. 過濾 LOO Error
        largest_3_indices = np.abs(loo_error).nlargest(3).index
        x_train_filtered_loo = x_train.drop(index=largest_3_indices)
        y_train_filtered_loo = y_train.drop(index=largest_3_indices)

        model3, _, _, rmse_train_loo = self._run_single_regression(x_train_filtered_loo, y_train_filtered_loo)
        rmse_test_loo = self._evaluate_on_test(model3, x_test, y_test)
        influence3 = model3.get_influence()
        leverage3 = influence3.hat_matrix_diag
        residuals3 = y_train_filtered_loo - model3.predict(x_train_filtered_loo)
        loo_error3 = residuals3 / (1 - leverage3)

        self.results["過濾 LOO Error"] = [
            f"{rmse_train_loo:.4f}", f"{rmse_test_loo:.4f}",
            f"{np.percentile(leverage3, [25, 50, 75])}", f"{np.max(leverage3):.4f}",
            f"{np.percentile(loo_error3, [25, 50, 75])}", f"{np.max(np.abs(loo_error3)):.4f}"
        ]

    def get_summary_report(self):
        """
        回傳最終的分析結果 DataFrame。
        """
        return pd.DataFrame(self.results)
