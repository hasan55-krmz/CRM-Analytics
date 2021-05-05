
#pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
pd.set_option('display.max_columns', None)

# Hemen verimizi hazırla


df_ = pd.read_excel(r"C:\Users\Erkan\Desktop\DSMLBC-4\4.Hafta_26-29_Ocak Haftası\Ödevler ve Çalışmalar\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

from helpers.helpers import check_df
check_df(df)

from helpers.helpers import crm_data_prep
df = crm_data_prep(df)
check_df(df)

df1=df.copy()

df_gm = df[df["Country"] =="Germany"]
df_gm.head()

df = df_gm.copy()

df.groupby(["Invoice","StockCode","TotalPrice"]).agg({"Quantity":"max","Price":"sum"}).head(20)

df.groupby(["Invoice","StockCode"]).agg({"Quantity":"sum"}).unstack().shape

df.StockCode.nunique()

df.groupby(["Invoice","StockCode"]).agg({"Quantity":"sum"}).unstack().fillna(0).applymap(lambda x: 1 if x >0 else 0).iloc[0:5, 0:5]

def create_invoicie_product_dataframe(dataframe):
    df = dataframe.groupby(["Invoice","StockCode"]).\
             agg({"Quantity":"sum"}).unstack().fillna(0).\
             applymap(lambda x: 1 if x >0 else 0)

    return df

df_inv_pro = create_invoicie_product_dataframe(df)

df_inv_pro.head()

# Çıtır ödev.
# Her bir invoice'da kaç eşsiz ürün vardır.

df_inv_pro["tot_product"] = df_inv_pro.apply(lambda x : x.sum(),axis = 1)

df_inv_pro.columns
# Her bir product kaç eşsiz sepettedir.

#df_inv_pro.sum()
df_inv_pro.loc["tot_sepet"] = df_inv_pro.apply(lambda x : x.sum(),axis = 0)
df_inv_pro.tail()
############################################
# Birliktelik Kurallarının Çıkarılması
############################################
df_inv_pro.drop("tot_product", axis = 1, inplace=True)

df_inv_pro.drop("tot_sepet",axis= 0, inplace =True)

frequent_itemsets = apriori(df_inv_pro, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("lift", ascending=False).head()


# tüm çalışmanın fonksiyonlaştırılması
import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import crm_data_prep, create_invoice_product_df

df_ = pd.read_excel(r"C:\Users\Erkan\Desktop\DSMLBC-4\4.Hafta_26-29_Ocak Haftası\Ödevler ve Çalışmalar\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df = crm_data_prep(df)

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    return rules

rules = create_rules(df)

