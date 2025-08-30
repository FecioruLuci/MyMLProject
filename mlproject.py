import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df = pd.read_csv("W:/vscode/Machine-Learning/MLPROJECT/bengaluru_house_prices.csv")
#print(df.head())

dfdrop = df.drop(["area_type","availability","society","balcony"],axis="columns")
#print(dfdrop)
#dfdrop["bath"] = dfdrop["bath"].fillna(dfdrop["bath"].mean())
df2 = dfdrop.dropna()
#print(df2.isnull().sum())
#print(df2["size"].unique())
df2["BHK"] = df2["size"].apply(lambda x: int(x.split(" ")[0]))
#print(df2[df2.BHK>20])
#43 bedrooms with 2400 sqfr when there is 8000 sqft with 27 bedrooms, makes no sense seems error

#print(df2.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

#print(df2[~df2["total_sqft"].apply(is_float)].head(20))

def convert(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df3 = df2.copy()
df3["total_sqft"] = df3["total_sqft"].apply(convert)

df4 = df3.copy()
#1 lahk = 100.000 rupees 
df4["price_per_sqft"] = df3.price*100000/df3.total_sqft
#print(len(df4.location.unique()))

df4.location = df4.location.apply(lambda x: x.strip())
location_stats = df4.groupby("location")["location"].agg("count").sort_values(ascending=False)
#print(location_stats)

other_loc = location_stats[location_stats<= 10]
df4.location = df4.location.apply(lambda x: "other" if x in other_loc.index else x)
print(df4.location.unique())
location_stats2 = df4.groupby("location")["location"].agg("count").sort_values(ascending=False)
print(location_stats2)
print(df4.head(10))