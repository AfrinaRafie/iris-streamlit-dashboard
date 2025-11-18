import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

st.title(" Iris Flower Data Visualization Dashboard")


# Load Dataset
@st.cache_data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["species"] = data.target
    df["species"] = df["species"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})
    return df

df = load_data()


# Sidebar Filters
st.sidebar.header("Filter Options")
species_filter = st.sidebar.selectbox("Select Species:", df["species"].unique())

filtered_df = df[df["species"] == species_filter]


# Data Summary
st.subheader(" Data Summary")
st.metric("Total Rows Selected", len(filtered_df))
st.dataframe(filtered_df)

# Visualization 1: Scatter Plot
st.subheader("Sepal Length vs Sepal Width")
fig1 = px.scatter(
    filtered_df,
    x="sepal length (cm)",
    y="sepal width (cm)",
    color="species",
    title="Sepal Length vs Sepal Width"
)
st.plotly_chart(fig1, use_container_width=True)

# Visualization 2: Histogram
st.subheader("Petal Length Distribution")
fig2 = px.histogram(
    filtered_df,
    x="petal length (cm)",
    color="species",
    title="Petal Length Histogram"
)
st.plotly_chart(fig2, use_container_width=True)


# Optional: Show Raw Data
if st.checkbox("Show Raw Iris Dataset"):
    st.write(df)