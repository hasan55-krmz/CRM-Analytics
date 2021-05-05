import pandas as pd

users = pd.read_csv("users.csv")
purchases  = pd.read_csv("purchases.csv")


df = purchases.merge(users, how="inner", on="uid")
df.shape
for i in ["country","device","gender","age"]:
    print(df[[i]].value_counts())


6*2*2*5
df.groupby(["country","device","gender","age"]).agg({"price":"sum"})
agg_df = df.groupby(["country","device","gender","age"]).agg({"price":"sum"}).sort_values("price", ascending=False)
agg_df.head()

agg_df.reset_index(inplace =True)
agg_df.head()


agg_df["age_cat"] = pd.cut(agg_df["age"], bins =[0,18,24,30,51,df["age"].max()],labels=["0_18","19_24","25_30","31_50","51+"])
agg_df.head()


agg_df["customer_lavel_based"] = [row[0]+"_"+row[1].upper()+"_"+row[2]+"_"+row[5] for row in agg_df.values]
agg_df.head()

agg_df = agg_df[["customer_lavel_based","price"]]
agg_df.head()

agg_df =agg_df.groupby("customer_lavel_based").agg({"price": "mean"})
agg_df.reset_index(inplace=True)
agg_df.head()


agg_df["segment"] = pd.qcut(agg_df["price"], 4,labels = ["D","C","B","A"])
agg_df.head()

agg_df.groupby("segment").agg({"price":"mean"})

agg_df.groupby("customer_lavel_based").agg({"price": "mean"})
userss= "TUR_IOS_F_0_18"
agg_df[agg_df["customer_lavel_based"] ==userss]

agg_df.shape