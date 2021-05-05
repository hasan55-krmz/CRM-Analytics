import pandas as pd

pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler


df_ = pd.read_excel("online_retail_II.xlsx",sheet_name="Year 2010-2011")
df = df_.copy()
df.head()


#############################################################
# Veri Ön İşleme
#############################################################


df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)

df["TotalPrice"] = df["Quantity"]*df["Price"]
df.head()

cltv_df = df.groupby("Customer ID").agg({"Invoice": lambda x:len(x),
                                         "Quantity": lambda x: x.sum(),
                                         "TotalPrice": lambda x : x.sum()})

cltv_df.columns = ['total_transaction', 'total_unit', 'total_price']
cltv_df.head()

#############################################################
# Hesaplamalar
#############################################################

cltv_df['avg_order_value'] = cltv_df["total_price"]/cltv_df["total_transaction"]
cltv_df["purchase_frequency"] = cltv_df["total_transaction"]/cltv_df.shape[0]

repeat_rate = cltv_df[cltv_df["total_transaction"]>1].shape[0]/cltv_df.shape[0]
churn_rate = 1-repeat_rate

cltv_df["profit_margin"] = cltv_df["total_price"]*0.05

# Customer Value
cltv_df["CV"] = cltv_df["avg_order_value"]*cltv_df["purchase_frequency"]

# Customer LT Value
cltv_df["CLTV"] = (cltv_df["CV"]/churn_rate)*cltv_df["profit_margin"]

cltv_df.head()

#############################################################
# Ölçeklendirme ve Segmentasyon
#############################################################

scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])

cltv_df.head()

cltv_df["segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_df[["segment", "total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV",ascending=False).head()

CLTV = cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg({"count", "mean", "sum"})

CLTV.to_excel("cltv.xlsx")
df["Customer ID"].nunique()
1085*3+1084

################################################################################
# Yorumlama
################################################################################
cltv_df.shape
df.shape

# Segmenlere göre df veri setini filtrelemek için fonksiyon
def cv_group(cv_group):
    indeksler =[df[df["Customer ID"] == i].index for i in cltv_df[cltv_df["segment"] == cv_group].index]
    son = []
    for i in range(len(indeksler)):
        for j in range(len(indeksler[i])):
            son.append(indeksler[i][j])
    df_cltv = df.loc[son]
    df_cltv["TotalPrice"] = df_cltv["Quantity"]*df_cltv["Price"]
    return df_cltv

cltv_A =cv_group("A")

cltv_B =cv_group("B")
cltv_C =cv_group("C")
cltv_D =cv_group("D")

cltv_A.shape[0]+cltv_B.shape[0]+cltv_C.shape[0]+cltv_D.shape[0]
df.shape

for i in [cltv_A ,cltv_B ,cltv_C ,cltv_D]:

cltv_A.groupby("StockCode").agg({"Invoice": "count","Quantity":"sum","Price":"sum"}).sort_values(by="Invoice", ascending = False)

def cv_segment_urun(segment):
    data1 = cv_group(segment)
    data2 = data1.groupby("StockCode").agg({"Invoice": "count","Quantity":"sum","TotalPrice":"sum"}).sort_values(by="TotalPrice", ascending = False)
    data2["Perc_tot_price"] = data2["TotalPrice"]/data2["TotalPrice"].sum()*100
    data2["Price_cumsum"] = data2["Perc_tot_price"].cumsum()
    return data2

seg_A = cv_segment_urun("A")