
######################################################################
# Gerekli kütüphaneler, fonksiyonlat ve veri setini import ettim
#######################################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel(r"C:\Users\Erkan\Desktop\DSMLBC-4\4.Hafta_26-29_Ocak Haftası\Ödevler ve Çalışmalar\online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

##########################################################################
# VEri Ön İşleme
##########################################################################

#Veriyi inceledim
df.shape
df.head()
df.tail()
df.info()
df.describe().T

#Verideki nan değerlerin oldğu gözlemleri attım
df.dropna(inplace=True)

# Veride C harfi içeren İnvoice'leirn buulunduğu gözlemleri attım. C harfi içereknelr iade olduğunu ifade etmekte
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0] # her işlemlde en az bir tane ürün almış olnları seçtim

df.describe().T

# UK müşterilerini seçiyorum
df = df[df["Country"] =="United Kingdom"]
df.head()

#Bu aşamaya kadar oluşan son veriyi inceleyelim
df.describe().T
plt.boxplot(df["Quantity"])
plt.boxplot(df["Price"])
plt.show()
# Aykırı değerleri tespit edip baskılama işlemleri için yazılan fonksiyonları uyguladım
replace_with_thresholds(df, "Quantity") # Miktarda büyük aykırı değerli maksimum olan değerler değiştirdi
replace_with_thresholds(df, "Price") #Price da büyük aykırı değerleri maksimum olarak belirlenen değerle değiştirdi
df.describe().T
plt.boxplot(df["Quantity"])
plt.boxplot(df["Price"])
plt.show()

df["TotalPrice"] = df["Quantity"] * df["Price"] # alınan ürünlerin toplam miktarı ile fiyatı çarpıldı. her bir üründe ne kadar para bıkaktı

df["InvoiceDate"].max() #Veri setinde en son işlem yapılan tarih
today_date = dt.datetime(2011, 12, 11)

df.head()
df.shape
##############################################
# RFM TABLE
###############################################


rfm  = df.groupby("Customer ID").agg({"InvoiceDate":[lambda date:(today_date-date.max()).days,lambda date:(today_date-date.min()).days],
                                      "Invoice": lambda x: x.nunique(),
                                      "TotalPrice":lambda  x: x.sum()})
rfm.head()
rfm.columns = rfm.columns.droplevel(0)
rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

rfm["monetary"] = rfm["monetary"] / rfm["frequency"] #Her brr müşteri alışveriş başına ortalma bıraktığı para

rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)

rfm["recency_weekly_p"] = rfm["recency_cltv_p"] / 7 #Recency değerini günlükten haftalığa çevirdim
rfm["T_weekly"] = rfm["T"] / 7 # Tenur değerini günlükten haftalığa çevirdim

rfm = rfm[rfm["monetary_avg"] > 0]
rfm = rfm[(rfm['frequency'] > 1)]
rfm["frequency"] = rfm["frequency"].astype(int) # Frewunecyi değerini inte çevirdim

##############################################################
# 2. BG/NBD Modelinin Kurulması
##############################################################

# pip install lifetimes

bgf = BetaGeoFitter(penalizer_coef=0.001) # bgf model nesnesini oluşturdum

# Frequency, Recency ve Tenur değişkenleri ile BGNBD  modelin eğitiyoruz
bgf.fit(rfm['frequency'], rfm['recency_weekly_p'], rfm['T_weekly'])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        rfm['frequency'],
                                                        rfm['recency_weekly_p'],
                                                        rfm['T_weekly']).sort_values(ascending=False).head(10)

################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################


m1=bgf.predict(4,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sort_values(ascending=False)


################################################################
# 1 Ay içinde tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################


bgf.predict(4 * 3,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()

# 1 aylık beklenen satış ile 3 aylık beklenen satışın grafiklerini çizdidim

rfm["exp_P_1"] = bgf.predict(4,rfm['frequency'],rfm['recency_weekly_p'],rfm['T_weekly'])
rfm["exp_P_3"] = bgf.predict(4*3,rfm['frequency'],rfm['recency_weekly_p'],rfm['T_weekly'])

fig, ax = plt.subplots()
ax.plot(rfm.index,rfm["exp_P_1"])
ax.plot(rfm.index,rfm["exp_P_3"])
plt.show()


##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması,
##############################################################
rfm.head()
ggf = GammaGammaFitter(penalizer_coef=0.01) # ggf model nesnesini oluşturdum
ggf.fit(rfm['frequency'], rfm['monetary_avg']) # veri setindeki frequency ve monetary_avg değişkenleri ile gama-gama moelini eğittim

# Gamam-Gama modeli ile beklenn ortalama getiriyi hesapladık!!!!!!!!!!!Burada Zamana göre prediction  yapamıyor muyuz? Ki zaten beklenen getiri zamanla değişmez mi?
ggf.conditional_expected_average_profit(rfm['frequency'],
                                           rfm['monetary_avg']).head(10)

ggf.conditional_expected_average_profit(rfm['frequency'],
                                        rfm['monetary_avg']).sort_values(ascending=False).head(10)

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                         rfm['monetary_avg'])

rfm.sort_values("expected_average_profit", ascending=False).head(20)

rfm.shape
##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

##############################################################
# GÖREV 1
##############################################################
# - 2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.
# - Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.
# - Mantıksız ya da çok isabetli olduğunu düşündüğünüz sonuçları vurgulayınız.
# - Dikkat! 6 aylık expected sales değil cltv prediction yapılmasını bekliyoruz.
#   Yani direk bgnbd ve gamma modellerini kurarak devam ediniz ve
# - cltv prediction için ay bölümüne 6 giriniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

rfm.shape
cltv.shape

cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
rfm_cltv_final = rfm.merge(cltv, on="Customer ID", how="left")
rfm_cltv_final.head()




##############################################################
# GÖREV 2
##############################################################
# - 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# - 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
# - Fark var mı? Varsa sizce neden olabilir?
# Dikkat! Sıfırdan model kurulmasına gerek yoktur.
# Var olan bgf ve ggf üzerinden direk cltv hesaplanabilir.

# 1 Aylık CLTV
cltv_1M  = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=1,
                                   freq="W",
                                   discount_rate=0.01)
cltv_1M = cltv_1M.reset_index()

# 12 Aylık CLTV
cltv_12M  = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv_12M = cltv_12M.reset_index()

# CLTV leri birleşir
rfm_cltv_1M_12M = rfm.merge(cltv_1M, on="Customer ID", how="left") # 1 aylık cltv yi rfm ile birleştirdim
rfm_cltv_1M_12M.rename(columns={"clv": "clv_1M"}, inplace=True) # clv olarak adlandırdığı 1 ayylık tahmini gösteren colon ismini clv_1M yaptım

rfm_cltv_1M_12M.head()

rfm_cltv_1M_12M = rfm_cltv_1M_12M.merge(cltv_12M, on ="Customer ID", how="left") #12 Aylık cltvyi de birleştirdim
rfm_cltv_1M_12M.rename(columns={"clv": "clv_12M"}, inplace=True) # clv olarak adlandırdığı 12 ayylık tahmini gösteren colon ismini clv_12M yaptım

rfm_cltv_1M_12M.head()

# 1 aylık cltv  en yüksek olanlar
rfm_cltv_1M_12M.sort_values(by="clv_1M", ascending =False).head(20)

# 12 aylık cltv en yüksek olanlar
rfm_cltv_1M_12M.sort_values(by="clv_12M", ascending =False).head(20)




# 12 aylık en yüksek clvye sahip müşterilerin 12 aylık ve 1 aylık clv lerinin grafiği
index_20=pd.DataFrame(range(0,20)) # 0-20 arasında df oluşturdum
fig, ax = plt.subplots()
ax.plot(index_20[0],rfm_cltv_1M_12M.sort_values(by="clv_12M", ascending =False)["clv_12M"].head(20))
ax.plot(index_20[0],rfm_cltv_1M_12M.sort_values(by="clv_1M", ascending =False)["clv_1M"].head(20))
plt.show()

rfm_cltv_1M_12M.sort_values(by="clv_12M", ascending =False)["clv_1M"].head(20)





##############################################################
# GÖREV 3
##############################################################
# 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 3 gruba (segmente) ayırınız ve
# grup isimlerini veri setine ekleyiniz. Örneğin (C, B, A)
# CLTV'ye göre en iyi yüzde 20'yi seçiniz. Ve bunlara top_flag yazınız. yüzde 20'ye 1.
# diğerlerine 0 yazınız.

# 3 grubu veri setindeki diğer değişkenler açısıdan analiz ediniz.
# 3 grup için yönetime 6 aylık aksiyon önerilerinde bulununuz. Kısa kısa.

cltv =ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv = cltv.reset_index()

rfm_cltv_final_3 =rfm.merge((cltv, on="Customer ID", how="left"))

rfm_cltv_final_3["segment"] = pd.qcut(rfm_cltv_final_3["clv"],3,labels=["C","B","A"])

idea.case.sensitive.fs=true
idea.case.sensitive.fs =True