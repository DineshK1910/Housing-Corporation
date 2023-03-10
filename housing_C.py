from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator,TransformerMixin
import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv("housing.csv")
print(df.head())
print(df.info())
print(df.sum().isnull())
print(df.dtypes)
df.plot(kind="scatter",x="longitude",y="latitude")
plt.savefig("bad visualization")
plt.show()
df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
plt.savefig("better visualization")
plt.show()
df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=df["population"]/100,label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,sharex=False)
plt.show()
import matplotlib.image as mpimg
cal=mpimg.imread("California.png")
ax=df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=df["population"]/100,label="population",c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.imshow(cal,extent=[-124.55,-113.80,32.45,42.05],alpha=0.5,cmap=plt.get_cmap("jet"))
plt.ylabel("latitude",fontsize=14)
plt.xlabel("longitude",fontsize=14)
prices=df["median_house_value"]
tick_val=np.linspace(prices.min(),prices.max(),11)
cbar=plt.colorbar(ticks=tick_val/prices.max())
cbar.ax.set_yticklabels(["$%dk"%round(v/1000) for v in tick_val],fontsize=14)
cbar.set_label("median_house_value",fontsize=16)
#plt.legend()
plt.savefig("cal_hp")
plt.show()
#CORRELATIONS
cr=df.corr()
print(cr["median_house_value"].sort_values(ascending=False))
from pandas.plotting import scatter_matrix
att=["median_house_value", "median_income", "total_rooms",
 "housing_median_age"]
scatter_matrix(df[att],figsize=(12,8))
plt.show()
#ATTRIBUTE COMBINATIONS
df["rooms_per_household"]=df["total_rooms"]/df["households"]
from sklearn.preprocessing import OneHotEncoder
oh=OneHotEncoder()
h=oh.fit_transform(df[["ocean_proximity"]])
print(h.toarray())
print(oh.categories_)
print(df.values)
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(strategy="median")
housing_n=df.drop("ocean_proximity",axis=1)
imputer.fit(housing_n)
print(imputer.statistics_)
X=imputer.transform(housing_n)
ht=pd.DataFrame(X,columns=housing_n.columns)
room_ix,bedroom_ix,population_ix,household_ix=3,4,5,6
class combattradder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedroom_per_room=True):
        self.add_bedroom_per_room=add_bedroom_per_room
    def fit(self,X,y=None):
        return(self)
    def transform(self,X,y=None):
        r_p_h=X[:,room_ix]/X[:,household_ix]
        p_p_h=X[:,population_ix]/X[:,household_ix]
        if self.add_bedroom_per_room:
            bpr=X[:,bedroom_ix]/X[:,room_ix]
            return(np.c_[X,r_p_h,bpr,p_p_h])
        else:
            return(np.c_[X,r_p_h,p_p_h])
attradder=combattradder(add_bedroom_per_room=False)
hattr=attradder.transform(df.values)
hattr=pd.DataFrame(X,columns=housing_n.columns)
y=df["median_house_value"]
xtr,xts,ytr,yts=train_test_split(hattr,y,test_size=0.3)
lr=LinearRegression()
lr.fit(xtr,ytr)
ypred=lr.predict(xts)
print("the accuracy:",accuracy_score(yts,ypred))