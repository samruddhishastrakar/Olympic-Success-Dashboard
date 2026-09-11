import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Olympic Success Analysis",
    page_icon="🏅",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("country_year_dashboard.csv")


@st.cache_resource
def load_model():
    return joblib.load("olympic_random_forest_model_compressed.pkl")


@st.cache_resource
def load_features():
    return joblib.load("olympic_feature_names.pkl")


df = load_data()
model = load_model()
feature_names = list(load_features())


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏅 Olympic Success Analysis")
st.subheader("Factors Behind Olympic Success Using Machine Learning")

st.markdown(
    """
    This interactive dashboard analyses Olympic performance using
    athlete characteristics, participation, medal outcomes and
    country-level economic indicators such as GDP and population.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Dashboard")
st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Olympic Overview",
        "Country & Economic Analysis",
        "Medal Prediction",
        "Model Insights"
    ]
)


# --------------------------------------------------
# OLYMPIC OVERVIEW
# --------------------------------------------------

if page == "Olympic Overview":

    st.header("Olympic Overview")

    total_records = df["Total_Athletes"].sum()
    total_medals = df["Total_Medals"].sum()
    total_countries = df["NOC"].nunique()
    total_years = df["Year"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Olympic Records",
            f"{int(total_records):,}"
        )

    with col2:
        st.metric(
            "Total Medals",
            f"{int(total_medals):,}"
        )

    with col3:
        st.metric(
            "Countries / NOCs",
            total_countries
        )

    with col4:
        st.metric(
            "Olympic Years",
            total_years
        )

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
        df.sort_values(
            "Total_Medals",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# COUNTRY & ECONOMIC ANALYSIS
# --------------------------------------------------

elif page == "Country & Economic Analysis":

    st.header("Country & Economic Analysis")

    countries = sorted(
        df["NOC"].dropna().unique()
    )

    selected_country = st.selectbox(
        "Select Country / NOC",
        countries
    )

    country_data = df[
        df["NOC"] == selected_country
    ].copy()

    if country_data.empty:

        st.warning(
            "No data available for this country."
        )

    else:

        years = sorted(
            country_data["Year"].unique()
        )

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
                [
                    "Year",
                    "Total_Athletes",
                    "Total_Medals"
                ]
            ].set_index("Year")

            st.line_chart(chart_data)

            st.subheader("Economic Indicators")

            economic_chart = country_data[
                [
                    "Year",
                    "GDP",
                    "Population"
                ]
            ].set_index("Year")

            st.line_chart(economic_chart)

            st.subheader(
                "Selected Country-Year Data"
            )

            st.dataframe(
                selected_data,
                use_container_width=True,
                hide_index=True
            )


# --------------------------------------------------
# MEDAL PREDICTION
# --------------------------------------------------

elif page == "Medal Prediction":

    st.header("🏅 Olympic Medal Prediction")

    st.markdown(
        """
        Enter athlete and country-level information below.
        The trained Random Forest machine learning model will
        predict whether the record is likely to result in a medal.
        """
    )

    st.divider()

    # Extract categories from trained model feature names

    noc_options = [
        x.replace("NOC_", "")
        for x in feature_names
        if x.startswith("NOC_")
    ]

    sport_options = [
        x.replace("Sport_", "")
        for x in feature_names
        if x.startswith("Sport_")
    ]

    region_options = [
        x.replace("region_", "")
        for x in feature_names
        if x.startswith("region_")
    ]

    sex_options = ["F", "M"]
    season_options = ["Summer", "Winter"]

    col1, col2 = st.columns(2)

    with col1:

        sex = st.selectbox(
            "Sex",
            sex_options
        )

        age = st.number_input(
            "Age",
            min_value=10.0,
            max_value=80.0,
            value=24.0,
            step=1.0
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=250.0,
            value=170.0,
            step=1.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=200.0,
            value=65.0,
            step=1.0
        )

        year = st.number_input(
            "Olympic Year",
            min_value=1896,
            max_value=2032,
            value=2020,
            step=4
        )

    with col2:

        noc = st.selectbox(
            "NOC",
            sorted(noc_options)
        )

        season = st.selectbox(
            "Season",
            season_options
        )

        sport = st.selectbox(
            "Sport",
            sorted(sport_options)
        )

        gdp = st.number_input(
            "GDP",
            min_value=0.0,
            value=100000000000.0,
            step=1000000000.0,
            format="%.0f"
        )

        population = st.number_input(
            "Population",
            min_value=0.0,
            value=50000000.0,
            step=1000000.0,
            format="%.0f"
        )

    region = st.selectbox(
        "Region",
        sorted(region_options)
    )

    st.divider()

    if st.button(
        "🔮 Predict Medal Outcome",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame({
            "Sex": [sex],
            "Age": [age],
            "Height": [height],
            "Weight": [weight],
            "NOC": [noc],
            "Year": [year],
            "Season": [season],
            "Sport": [sport],
            "GDP": [gdp],
            "Population": [population],
            "region": [region]
        })

        categorical_features = [
            "Sex",
            "NOC",
            "Season",
            "Sport",
            "region"
        ]

        input_encoded = pd.get_dummies(
            input_data,
            columns=categorical_features,
            drop_first=True
        )

        input_encoded = input_encoded.reindex(
            columns=feature_names,
            fill_value=0
        )

        prediction = model.predict(
            input_encoded
        )[0]

        probability = model.predict_proba(
            input_encoded
        )[0][1]

        st.divider()

        if prediction == 1:

            st.success(
                "🏅 Prediction: Medal Won"
            )

        else:

            st.info(
                "Prediction: No Medal"
            )

        st.metric(
            "Probability of Medal",
            f"{probability * 100:.2f}%"
        )

        st.caption(
            "This prediction represents the output of the trained "
            "Random Forest model and should not be interpreted as "
            "a guarantee of Olympic success."
        )


# --------------------------------------------------
# MODEL INSIGHTS
# --------------------------------------------------

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

    st.subheader(
        "Model Performance Comparison"
    )

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

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Random Forest ROC-AUC",
            "0.871"
        )

    with col2:
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
        "Feature importance indicates the relative contribution "
        "of variables to the Random Forest model and should not "
        "be interpreted as proof of causation."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Research Project: Factors Behind Olympic Success Using Machine Learning"
)
