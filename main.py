import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_and_transform_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
    df = pd.read_excel(url)
    df.columns = ['No', 'Date', 'Age', 'Dist_MRT', 'Stores', 'Lat', 'Long', 'Price']

    df['Dist_Center'] = np.sqrt((df['Lat']) + (df['Long']))

    cols = [c for c in df.columns if c != 'Price'] + ['Price']
    df = df[cols]

    return df.drop(columns=['No', 'Date', 'Lat', 'Long'])


def perform_eda(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn',center=0, fmt=".2f")
    plt.title("Корелация на данни")
    plt.show()


    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.regplot(x='Dist_MRT', y='Price', data=df, ax=axes[0, 0],
                scatter_kws={'alpha': 0.4, 'color': 'teal'}, line_kws={'color': 'red'})
    axes[0, 0].set_title('Разстояние до метро vs Цена')

    # Възраст на сградата
    sns.regplot(x='Age', y='Price', data=df, ax=axes[0, 1], order=2,
                scatter_kws={'alpha': 0.4, 'color': 'orange'}, line_kws={'color': 'red'})
    axes[0, 1].set_title('Възраст на сградата vs Цена')

    # Брой магазини
    sns.boxplot(x='Stores', y='Price', data=df, ax=axes[1, 0],
                hue='Stores', palette='viridis', legend=False)
    axes[1, 0].set_title('Брой магазини vs Цена')

    # Разстояние до центъра
    sns.regplot(x='Dist_Center', y='Price', data=df, ax=axes[1, 1],
                scatter_kws={'alpha': 0.4, 'color': 'purple'}, line_kws={'color': 'red'})
    axes[1, 1].set_title('Разстояние до центъра vs Цена')

    plt.tight_layout()
    plt.show()



def build_linear_model(df):
    X = df[['Age', 'Dist_MRT', 'Stores']]
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("\n--- СКАЛИРАНИ КОЕФИЦИЕНТИ ---")

    predictions = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print('Mean Squared Error:', mse.__round__(3))
    print('R-squared:', r2.__round__(3))
    print('Mean Absolute Error:', mae.__round__(3))
    print('Root Mean Squared Error:', rmse.__round__(3))

    residuals = y_test - predictions
    plt.scatter(predictions, residuals)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Analysis")
    plt.show()

    # Къде моделът греши
    feature_importance = np.abs(model.coef_)
    feature_names = X.columns
    sorted_indices = np.argsort(feature_importance)

    # сортиране по важноста
    plt.barh(range(len(feature_importance)), feature_importance[sorted_indices], align='center')
    plt.yticks(range(len(feature_importance)), feature_names[sorted_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Linear Regression Feature Importance')
    plt.show()

    # График на линейна регресия
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Predictions vs Actuals')
    z = np.polyfit(y_test, predictions, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), color='magenta')
    plt.show()

    return model,scaler

def run_statistics(df):
    print("\n--- СТАТИСТИЧЕСКИЕ ДОКАЗАТЕЛЬСТВА ---")
    factors = ['Dist_MRT', 'Stores', 'Age', 'Dist_Center']

    for factor in factors:
        r_val, p_val = stats.pearsonr(df[factor], df['Price'])

        significance = "Значим" if p_val < 0.05 else "Незначим"

        print(f"Фактор [{factor:11}]: Корреляция = {r_val:6.3f} | p-value = {p_val:.5f} | Статус: {significance}")


def predict_price(age, dist_mrt, stores, model, scaler):
    input_data = pd.DataFrame(
        [[age, dist_mrt, stores]],
        columns=['Age', 'Dist_MRT', 'Stores']
    )

    input_scaled = scaler.transform(input_data)
    predicted = model.predict(input_scaled)

    return predicted[0]


def compare_models_and_fix_collinearity(df):
    print("\nДИАГНОСТИКА И ОПТИМИЗАЦИЯ НА МОДЕЛА")

    features_A = ['Age', 'Dist_MRT', 'Stores', 'Dist_Center']
    features_B = ['Age', 'Dist_MRT', 'Stores']

    y = df['Price']

    # Изчисляване на VIF
    X_vif = df[features_A]
    vif_data = pd.DataFrame()
    vif_data["Фактор"] = X_vif.columns
    vif_data["VIF индекс"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]

    print("\nПроверка за VIF")
    print(vif_data)
    print("VIF > 10 показва сериозна зависимост между факторите")

    # Сравнително обучение
    results = []

    for name, feat_list in [("Модел А", features_A), ("Модел B", features_B)]:
        X = df[feat_list]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        results.append({"Модел": name, "R2": round(r2, 4), "MAE": round(mae, 4)})

    # Показване на сравнението
    print("\nСравнение на точността:")
    results_df = pd.DataFrame(results)
    print(results_df)

    return results_df


if __name__ == "__main__":
    df = load_and_transform_data()
    run_statistics(df)
    perform_eda(df)

    model, scaler = build_linear_model(df)

    compare_models_and_fix_collinearity(df)
    price = predict_price(
        age=16.2,
        dist_mrt=289.3248,
        stores=5,
        model=model,
        scaler=scaler
    )

    print(f"\nПрогнозирана цена: {price:.2f}")