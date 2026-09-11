import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Olympic Success Analysis",
    page_icon="🏅",
    layout="wide"
)

# -----------------------------
# Load Dashboard Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("country_year_dashboard.csv")

df = load_data()

# -----------------------------
# Page Header
# -----------------------------
st.title("🏅 Olympic Success Analysis")
st.subheader("Factors Behind Olympic Success Using Machine Learning")

st.markdown(
    """
    This interactive dashboard analyses Olympic performance using
    athlete participation, medal outcomes and country-level economic
    indicators such as GDP and population.
    """
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Dashboard")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Olympic Overview",
        "Country & Economic Analysis",
        "Model Insights"
    ]
)

# -----------------------------
# Olympic Overview
# -----------------------------
if page == "Olympic Overview":

    st.header("Olympic Overview")

    total_records = df["Total_Athletes"].sum()
    total_medals = df["Total_Medals"].sum()
    total_countries = df["NOC"].nunique()
    total_years = df["Year"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Olympic Records", f"{int(total_records):,}")

    with col2:
        st.metric("Total Medals", f"{int(total_medals):,}")

    with col3:
        st.metric("Countries / NOCs", f"{total_countries}")

    with col4:
        st.metric("Olympic Years", f"{total_years}")

    st.divider()

    st.subheader("Top 10 Countries by Total Medals")

    top_countries = (
        df.groupby("NOC")["Total_Medals"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    st.bar_chart(
        top_countries.set_index("NOC")
    )

    st.subheader("Country-Year Dataset")

    st.dataframe(
        df.sort_values("Total_Medals", ascending=False),
        use_container_width=True,
        hide_index=True
    )


# -----------------------------
# Country & Economic Analysis
# -----------------------------
elif page == "Country & Economic Analysis":

    st.header("Country & Economic Analysis")

    countries = sorted(df["NOC"].dropna().unique())

    selected_country = st.selectbox(
        "Select Country / NOC",
        countries
    )

    country_data = df[df["NOC"] == selected_country].copy()

    if country_data.empty:
        st.warning("No data available for this country.")
    else:

        years = sorted(country_data["Year"].unique())

        selected_year = st.selectbox(
            "Select Olympic Year",
            years
        )

        selected_data = country_data[
            country_data["Year"] == selected_year
        ]

        if not selected_data.empty:

            row = selected_data.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Athletes",
                    int(row["Total_Athletes"])
                )

            with col2:
                st.metric(
                    "Total Medals",
                    int(row["Total_Medals"])
                )

            with col3:
                st.metric(
                    "GDP",
                    f"${row['GDP']:,.0f}"
                )

            with col4:
                st.metric(
                    "Population",
                    f"{row['Population']:,.0f}"
                )

            st.divider()

            st.subheader(
                f"{selected_country} — Olympic Performance"
            )

            chart_data = country_data[
                ["Year", "Total_Athletes", "Total_Medals"]
            ].set_index("Year")

            st.line_chart(chart_data)

            st.subheader("Economic Indicators")

            economic_chart = country_data[
                ["Year", "GDP", "Population"]
            ].set_index("Year")

            st.line_chart(economic_chart)

            st.subheader("Selected Country-Year Data")

            st.dataframe(
                selected_data,
                use_container_width=True,
                hide_index=True
            )


# -----------------------------
# Model Insights
# -----------------------------
elif page == "Model Insights":

    st.header("Machine Learning Model Insights")

    st.markdown(
        """
        Three machine learning models were evaluated for predicting
        whether an Olympic athlete/event record resulted in a medal.
        """
    )

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree"
        ],
        "Accuracy": [
            0.147,
            0.878,
            0.594
        ],
        "Precision": [
            0.147,
            0.591,
            0.242
        ],
        "Recall": [
            1.000,
            0.554,
            0.832
        ],
        "F1-Score": [
            0.256,
            0.571,
            0.375
        ],
        "ROC-AUC": [
            0.592,
            0.871,
            0.773
        ]
    })

    st.subheader("Model Performance Comparison")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Selected Model")

    st.success(
        "Random Forest was selected as the best-performing model "
        "based on its overall performance and highest ROC-AUC."
    )

    st.metric(
        "Random Forest ROC-AUC",
        "0.871"
    )

    st.metric(
        "Random Forest Accuracy",
        "87.8%"
    )

    st.divider()

    st.subheader("Top Features")

    importance = pd.DataFrame({
        "Feature": [
            "Year",
            "Age",
            "Weight",
            "Height",
            "Population",
            "GDP",
            "Sex",
            "Sport",
            "Region"
        ],
        "Importance": [
            0.1254,
            0.1243,
            0.0983,
            0.0943,
            0.0596,
            0.0578,
            0.0174,
            0.0119,
            0.0117
        ]
    })

    st.bar_chart(
        importance.set_index("Feature")
    )

    st.caption(
        "Feature importance indicates the relative contribution of "
        "variables to the Random Forest model and should not be "
        "interpreted as proof of causation."
    )

st.divider()

st.caption(
    "Research Project: Factors Behind Olympic Success Using Machine Learning"
)
