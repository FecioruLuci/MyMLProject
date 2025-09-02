import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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
#print(df4.location.unique())
location_stats2 = df4.groupby("location")["location"].agg("count").sort_values(ascending=False)
#print(location_stats2)
#print(df4.head(10))




#PART 2




#print(df4[df4.total_sqft/df4.BHK < 300].head(10))

df5 = df4[~(df4.total_sqft/df4.BHK < 300)]
#print(df5.shape)

def remove_outliner(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft <= (m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df6 = remove_outliner(df5)
#print(df6.shape)

def plotul(df,location):
    bhk2 = df[(df.location==location) & (df.BHK == 2)]
    bhk3 = df[(df.location == location) & (df.BHK == 3)]
    plt.scatter(bhk2.total_sqft, bhk2.price,color="blue", label= "2BHK", s=50)
    plt.scatter(bhk3.total_sqft, bhk3.price, color="green", marker="+", label="3BHK",s=50)
    plt.xlabel("total_sqft_area")
    plt.ylabel("price")
    plt.title(location)
    plt.legend()
    plt.show()

#plotul(df6,"Rajaji Nagar")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
# df8 = df7.copy()
#print(df7.shape)

#plotul(df7,"Rajaji Nagar")


#PART 3



plt.hist(df7.price_per_sqft,rwidth=0.7)
plt.xlabel("Price per sqft")
plt.ylabel("Count")
#plt.show()

#print(df7[df7.bath>10])
#print(df7[df7.bath > df7.BHK+2])

df8 = df7[df7.bath < df7.BHK+2]
#print(df8.shape)
df9 = df8.drop(["size","price_per_sqft"],axis="columns")
#print(df9.head())

dummies = pd.get_dummies(df9.location)
#print(dummies)

df10 = pd.concat([df9,dummies],axis="columns")
df11 = df10.drop(["other"],axis="columns")
#print(df11)
df12 = df11.drop("location",axis="columns")
#print(df12.head(3))

x = df12.drop("price",axis="columns")
#print(x)
y = df12.price

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
