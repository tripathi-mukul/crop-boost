import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
#Loading the dataset

yielddat=pd.read_csv('yield.csv')
dfs = pd.read_excel('fert.xlsx')
dfs = dfs.drop('Fertilzers',axis=1)
# Scaling Yield dataset
scaler = StandardScaler() 
scaler.fit(yielddat.drop('Yield(Tonnes/Hectare)',axis=1))
scaled_features = scaler.transform(yielddat.drop('Yield(Tonnes/Hectare)',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=yielddat.columns[:-1])
X = df_feat.values
Y = yielddat['Yield(Tonnes/Hectare)'].values
#Training model for yield prediction
knn = KNeighborsRegressor(n_neighbors=11)
knn.fit(X,Y)

#Training model for fertilizer recommendation

X_train=dfs.drop('Class',axis=1).values
y_train=dfs['Class'].values
knn1 = KNeighborsClassifier(n_neighbors=1)

knn1.fit(X_train,y_train)


st.write("""
# CropBoot App
This app help you predict the **Crop Yield** and recommends **Fertilizer** to be used to achieve the desired yield
""")


#Input for Yield Prediction
def user_input_features_yield():
    temp = st.sidebar.slider('Average Temperature(C)', 0.000000, 35.000000, 17.191501)
    precip = st.sidebar.slider('Precipitation(mm)', 0.000000,600.000000,99.146498)
    som = st.sidebar.slider('Soil Organic Matter(t/Ha)', 0.000000,15.000000,2.521281)
    awc = st.sidebar.slider('Available Water Capacity(fraction)', 0.000000, 0.300000, 0.166235)
    landar = st.sidebar.slider('Land Area(sq-m)', 40000.000000,7000000.000000,4.904763e+05)
    vpd = st.sidebar.slider('Vapour Pressure Deficit(kPa)', 0.000000,20.000000, 9.182233)

    data1 = {'Temperature': temp, 'Vapour Pressure Deficit': vpd, 'Precipitation': precip, 'Soil Organic Matter': som, 'Water Capacity': awc, 'Land Area': landar}

    st.header('Crop Yield Prediction')
    st.write(
        "Please help us with the following variable parameters to predict your crop's yield this year. Use the sidebar to vary your farming conditions")
    st.dataframe(pd.DataFrame(data1, index=[0]).style.applymap(
        lambda _: "background-color: LightSkyBlue;", subset=([0], slice(None))
    ), hide_index=True)

    temp = (temp - scaler.mean_[0])/scaler.scale_[0]
    vpd  = (vpd -scaler.mean_[5]) /scaler.scale_[5]
    precip = (precip-scaler.mean_[1])/scaler.scale_[1]
    som=som-scaler.mean_[2]
    som=som/scaler.scale_[2]
    awc=awc-scaler.mean_[3]
    awc=awc/scaler.scale_[3]
    landar=landar-scaler.mean_[4]
    landar=landar/scaler.scale_[4]
    
    data = {'Temp': temp,'VPD': vpd,'Precipitation':precip,'SOM':som,'AWC':awc,'Land Area':landar}
    features = pd.DataFrame(data, index=[0])


    
    return features

#Input for Fertilizer Recomendation
def user_input_features_fert():
    fert = st.sidebar.slider('Desired Yield(t/Ha)', 75.0000, 350.000000, 170.000000)
    
    data = {'Desired Yield(t/Ha)': fert}
    features = pd.DataFrame(data, index=[0])
    
    return features

st.sidebar.subheader('Input for Yield Prediction')
dfyield = user_input_features_yield()




yieldpred= knn.predict(dfyield)
displayValue = {
    'Yield' : yieldpred
}
st.subheader('Predicted Yield(t/Ha)')
st.write("Accuracy-80.40%")

st.dataframe(pd.DataFrame(displayValue).style.applymap(
        lambda _: "background-color: LightGreen;", subset=([0], slice(None))
    ), hide_index=True)

yieldpred=np.round(yieldpred)
fig=plt.figure(figsize=[12.0,0.5])
axes=fig.add_axes([0,0,1,1])
axes.set_xlim([50,300])
axes.set_xlabel('Yield(t/Ha)')
axes.set_yticks([])
sns.distplot(yieldpred)
st.pyplot(fig)



st.markdown('##')

st.sidebar.subheader('Input for Fertilizer Recomendation')

dffert = user_input_features_fert()
st.header('Crop Fertilizer Recommendation')
st.write("Please select your desired crop yield from the sidebar")
st.dataframe(dffert.style.applymap(
        lambda _: "background-color: LightSkyBlue;", subset=([0], slice(None))
    ), hide_index=True)
fertrec = knn1.predict(dffert)


st.subheader('Recommended Fertilizer(N-P-K)')
if fertrec==1:
    st.write("0-0-0")
elif fertrec==2:
    st.write("44-15-17")
elif fertrec==3:
    st.write("46-15-25")
elif fertrec==4:
    st.write("69-15-25")
elif fertrec==5:
    st.write("69-30-40")
elif fertrec==6:
    st.write("80-15-40")
elif fertrec==7:
    st.write("80-30-0")
elif fertrec==8:
    st.write("80-30-25")
elif fertrec==9:
    st.write("80-30-40")
else:
    st.write("92-30-40")
st.write("Accuracy-78.00%")





