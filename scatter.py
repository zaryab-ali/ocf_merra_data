import xarray as xr
import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ds = xr.open_zarr("PATH TO ZARR MERRA DATA")

aerosol_ds = ds[['ALBEDO', 'DUSMASS', 'TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU']]

aerosol_df = aerosol_ds.mean(dim=["lat", "lon"]).to_dataframe().reset_index()

aerosol_df = aerosol_df.set_index('time')


pv_df = pd.read_csv("india_pv_history_2024.csv", parse_dates=['end_utc'])
pv_df = pv_df.rename(columns={'end_utc': 'time'})
pv_df = pv_df[['time', 'generation_power_mw']].set_index('time')

# Ensure aerosol data is sorted and has a datetime index
aerosol_df = aerosol_df.sort_index()

# Join using asof merge
pv_df = pv_df.sort_index()
merged = pd.merge_asof(pv_df, aerosol_df, left_index=True, right_index=True, direction='backward')

print(merged.corr()['generation_power_mw'])



def monthly_coefficient():
    merged1 = merged.copy()
    merged1.index = pd.to_datetime(merged1.index)

    # Group by month and compute Pearson correlation per group
    monthly_corr = (
        merged1.groupby(merged1.index.month)
        .apply(lambda df: df['generation_power_mw'].corr(df['TOTSCATAU']))
    )

    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    monthly_corr.index = monthly_corr.index.map(month_names)


    plt.figure(figsize=(10, 5))
    monthly_corr.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Monthly Pearson Correlation: PV vs TOTSCATAU', fontsize=14)
    plt.ylabel('Correlation Coefficient')
    plt.xlabel('Month')
    plt.ylim(-1, 1)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()



def scatter_plot():
    sns.set_theme(style="whitegrid")

    for col in ['ALBEDO', 'DUSMASS', 'TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU']:
        plt.figure(figsize=(7, 4))
        sns.scatterplot(
            data=merged,
            x=col,
            y='generation_power_mw',
            s=10,
            alpha=0.3,
            color='steelblue'
        )
        sns.regplot(  #trend line
            data=merged,
            x=col,
            y='generation_power_mw',
            scatter=False,
            color='darkred',
            line_kws={'linewidth': 1.5}
        )
        plt.title(f'PV vs {col}', fontsize=14)
        plt.xlabel(col)
        plt.ylabel('Generation Power (MW)')
        plt.tight_layout()
        plt.show()

def yearly_coefficient_graph():
    rolling_corr = merged['generation_power_mw'].rolling(168).corr(merged['TOTSCATAU'])
    rolling_corr.plot(title="Rolling Correlation with TOTSCATAU (7-day window)")
    plt.show()


def linear_regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    import pandas as pd

    # Drop NaNs just in case
    data = merged.dropna()

    # Features (aerosol variables)
    X = data[['ALBEDO', 'DUSMASS', 'TOTANGSTR', 'TOTEXTTAU', 'TOTSCATAU']]

    # Target (PV generation)
    y = data['generation_power_mw']

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predictions and metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Print results
    print("Linear Regression Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")

    print(f"\nIntercept: {model.intercept_:.4f}")
    print(f"R^2 Score: {r2:.4f}")


    # Scatter plot of actual vs predicted
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.4, s=15, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Ideal Fit')

    plt.title('Actual vs Predicted PV Generation')
    plt.xlabel('Actual Generation Power (MW)')
    plt.ylabel('Predicted Generation Power (MW)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


linear_regression()