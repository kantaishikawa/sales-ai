import streamlit as st
import pandas as pd
import csv
import os
from sklearn.linear_model import LogisticRegression

st.title("営業AIダッシュボード")

file = "sales_data.csv"

if not os.path.exists(file):
    df = pd.DataFrame(columns=["visit_reason","test_drive","discount","sale"])
    df.to_csv(file,index=False)

df = pd.read_csv(file)

st.header("営業データ入力")

visit = st.selectbox(
"来店理由",
["新規","既存","指名"]
)

test_drive = st.selectbox(
"試乗",
[0,1]
)

discount = st.selectbox(
"値引き",
["10万以下","10-20万","20-30万","30万以上"]
)

sale = st.selectbox(
"成約",
[0,1]
)

if st.button("保存"):

    with open(file,"a") as f:
        writer = csv.writer(f)
        writer.writerow([visit,test_drive,discount,sale])

    st.success("保存しました")

st.header("営業データ")

st.dataframe(df)

st.header("成約率")

if len(df)>0:

    rate = df["sale"].mean()

    st.write(f"成約率 {round(rate*100,1)} %")

st.header("AI分析")

if len(df)>5:

    df_ml = pd.get_dummies(df)

    X = df_ml.drop("sale",axis=1)
    y = df_ml["sale"]

    model = LogisticRegression(max_iter=1000)

    model.fit(X,y)

    importance = pd.Series(
        model.coef_[0],
        index=X.columns
    ).sort_values(ascending=False)

    st.write("成約要因ランキング")

    st.write(importance.head(10))
