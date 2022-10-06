import streamlit as st
import pickle
import numpy as np
import pandas as pd


def load_model():

    with open('saved_model.pkl','rb') as file:

        data = pickle.load(file)
        return data

data = load_model()

regressor = data['model']
lf_out_size = data['lf_out_size']
lf_out_type = data['lf_out_type']
lf_fat  = data["lf_fat"]
lf_out_loc = data["lf_out_loc"]
lf_type = data["lf_type"]
lf_out_id = data['lf_out_id']





def show_predict_page():

    st.title("Bigmart Sales Prediction")
    st.write('''### We need some information in order to predict sales  ''')


    

    size = {
    'Medium',
    'High',
    'Small'
    }

    out_type = {

    'Supermarket Type1',
    'Supermarket Type2',
    'Supermarket Type3',
    'Grocery Store'

    }

    fat = {
        'Low Fat','Regular'
    }

    out_loc = {
        'Tier 1', 'Tier 2', 'Tier 3'
    }

    item_type = {
        'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables',
        'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
        'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
        'Breads', 'Starchy Foods', 'Others', 'Seafood'
    }

    out_id = {
        'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045',
        'OUT017', 'OUT046', 'OUT035', 'OUT019'
    }
    est_year = {
        1999, 2009, 1998, 1987, 1985, 2002, 2007, 1997, 2004
    }


    a = st.selectbox('Item Fat Content',fat)
    b = st.selectbox("Item Type" , item_type)
    c = st.selectbox("Outlet Identifier" , out_id)
    d = st.selectbox("Establishment Year" , est_year)
    e = st.selectbox('Outlet Size',size)
    f = st.selectbox("Outlet Type" , out_type)
    g = st.selectbox("Outlet Location" , out_loc)

    ok = st.button("Calculate Sales")
    if ok:
        
        if a=='Low Fat':
            a='LF'
        else:
            a='R'

        X = np.array([[a,b,c,d,e,f,g]])

        X[:,4] = lf_out_size.transform(X[:,4])
        X[:,5] = lf_out_type.transform(X[:,5])
        X[:,0] = lf_fat.transform(X[:,0])
        X[:,6] = lf_out_loc.transform(X[:,6])
        X[:,1] = lf_type.transform(X[:,1])
        X[:,2] = lf_out_id.transform(X[:,2])

        X = X.astype(float)

        salary = regressor.predict(X)

        st.write(f"The Estimated Sales will be : ${salary[0]:.2f}")

        # st.plotly_chart






    
