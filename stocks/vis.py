import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", palette='deep')
df = pd.read_csv("data/AAPL.csv", index_col=0)  
S0 = df.iloc[:, 0].mean()
final = df.iloc[:, -1]

def var(stocks, S0, final): 
    final *= stocks
    final -= (S0 * stocks)
    idx95 = int(len(final) * (1 - .95))
    idx99 = int(len(final) * (1 - .99))
    cutoff95 = final.to_numpy()[idx95]
    cutoff99 = final.to_numpy()[idx99]
    print(f"VaR at a 95% Confidence Rate: {cutoff95:.2f}")
    print(f"VaR at a 99% Confidence Rate: {cutoff99:.2f}")

def bell(stock, S0, df): 
    df *= stock 
    df -= (S0 * stock)
    counts, edges = np.histogram(df.dropna(), bins=50)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    colors = ['#d95f02' if x < 0 else '#1b9e77' for x in bin_centers]

    plt.bar(bin_centers, counts, width=np.diff(edges), color=colors, edgecolor='white', alpha=0.8)

    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5,)
    plt.axvline(-9279, color='red', linestyle='-', linewidth=1, alpha=0.5, label="95% VaR")
    plt.axvline(-18741, color='red', linestyle='--', linewidth=1, alpha=0.5, label="99% VaR")

    plt.legend(loc='upper left') 
 
    plt.title("VaR of AAPL Over 1 Year (200 Stocks)")
    plt.xlabel("Portfolio Profit / Loss")
    plt.ylabel("Frequency of Trials")

    plt.savefig('conditional_histogram.png', bbox_inches='tight')
    print("Plot successfully saved to 'conditional_histogram.png'!")    

def project(df):
    sub = df.sample(1000, random_state=0)

    df_transposed = sub.T

    sns.lineplot(data=df_transposed, dashes=False, legend=False, palette="dark")
    step = 30 
    ticks = range(0, len(df_transposed.index), step)
    labels = [i for i in ticks]
    plt.xticks(ticks=ticks, labels=labels, rotation=45) 

    plt.title("Projected Price of AAPL Over 1 Year ")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    plt.savefig('my_line_plot.png', dpi=300, bbox_inches='tight')