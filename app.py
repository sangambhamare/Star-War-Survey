import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Star Wars Survey Dashboard",
    layout="wide"
)
st.title("Star Wars Survey: Data Cleaning, Visualization & Analysis")

# -------------------------
# Helper Functions
# -------------------------
def get_df_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing the input DataFrame's info.
    Columns include: Column name, Non-Null Count, Null Count, % Missing, Data Type, and Memory Usage.
    """
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "% Missing": (df.isnull().sum().values / len(df)) * 100,
        "Data Type": df.dtypes.astype(str).values
    })
    mem_usage = df.memory_usage(deep=True)[df.columns]
    info_df["Memory Usage (Bytes)"] = mem_usage.values
    return info_df

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file. First tries a tab delimiter; if only one column is found, it retries using a comma.
    (Caching has been removed to avoid using pyarrow.)
    """
    try:
        df = pd.read_csv(filepath, delimiter="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, delimiter=",")
    except Exception as e:
        st.error(f"Error reading file '{filepath}': {e}")
        return None
    return df

# -------------------------
# Load Data
# -------------------------
data_path = "star_wars.csv"
df_raw = load_data(data_path)
if df_raw is None:
    st.stop()

# Work on a copy for cleaning and further analysis
df_clean = df_raw.copy()

# -------------------------
# Main Tabs
# -------------------------
main_tab_labels = [
    "Data Cleaning",
    "Basic Visualization",
    "Basic Statistical Reports",
    "Enhanced Dashboard & Export",
    "Geospatial Visualization",
    "User Guide"
]
main_tabs = st.tabs(main_tab_labels)

# =========================
# TAB 1: Data Cleaning
# =========================
with main_tabs[0]:
    st.header("Data Cleaning")
    cleaning_tabs = st.tabs([
        "Original Data",
        "Drop Unwanted Columns",
        "Rename Columns",
        "Handle Missing Data",
        "Final Cleaned Data"
    ])
    # --- Original Data ---
    with cleaning_tabs[0]:
        st.subheader("Original Data (First 10 Rows)")
        st.dataframe(df_raw.head(10))
        st.markdown("#### Data Information")
        st.dataframe(get_df_info(df_raw))
    # --- Drop Unwanted Columns ---
    with cleaning_tabs[1]:
        st.subheader("Drop Unwanted Columns")
        unnamed_cols = [col for col in df_clean.columns if col.startswith("Unnamed")]
        st.write("Columns to drop:", unnamed_cols)
        df_clean = df_clean.drop(columns=unnamed_cols)
        st.dataframe(df_clean.head(10))
    # --- Rename Columns ---
    with cleaning_tabs[2]:
        st.subheader("Rename Columns")
        rename_mapping = {
            "Have you seen any of the 6 films in the Star Wars franchise?": "seen_films",
            "Do you consider yourself to be a fan of the Star Wars film franchise?": "is_fan",
            "Which of the following Star Wars films have you seen? Please select all that apply.": "films_seen",
            "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "film_ranking",
            "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "character_opinions",
        }
        st.write("Renaming using the following mapping:")
        st.write(rename_mapping)
        df_clean = df_clean.rename(columns={k: v for k, v in rename_mapping.items() if k in df_clean.columns})
        st.dataframe(df_clean.head(10))
    # --- Handle Missing Data ---
    with cleaning_tabs[3]:
        st.subheader("Handle Missing Data")
        st.markdown("##### Missing Data Counts")
        missing_counts = df_clean.isna().sum()
        st.dataframe(missing_counts.to_frame("Missing Count"))
        st.write("Filling missing values:")
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna("Not Specified")
        st.dataframe(df_clean.head(10))
    # --- Final Cleaned Data ---
    with cleaning_tabs[4]:
        st.subheader("Final Cleaned Data (Preview)")
        # Convert all object columns to string to ensure compatibility
        for col in df_clean.select_dtypes(include=["object"]).columns:
            df_clean[col] = df_clean[col].astype("string")
        st.dataframe(df_clean.head(10))
        st.markdown("#### Data Information")
        st.dataframe(get_df_info(df_clean))

# =========================
# TAB 2: Basic Visualization
# =========================
with main_tabs[1]:
    st.header("Basic Data Visualization")
    vis_tabs = st.tabs([
        "Fan Analysis",
        "Film Viewing",
        "Demographics",
        "Film Ranking",
        "Character Opinions"
    ])
    # --- Fan Analysis ---
    with vis_tabs[0]:
        st.subheader("Fan Analysis")
        if 'is_fan' in df_clean.columns:
            fan_counts = df_clean['is_fan'].value_counts().reset_index()
            fan_counts.columns = ['is_fan', 'count']
            chart = alt.Chart(fan_counts).mark_bar().encode(
                x=alt.X('is_fan:N', title='Fan Status'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['is_fan', 'count']
            ).properties(title="Fan Status Distribution", width=600, height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Column 'is_fan' not found.")
    # --- Film Viewing ---
    with vis_tabs[1]:
        st.subheader("Film Viewing Analysis")
        if 'seen_films' in df_clean.columns:
            seen_counts = df_clean['seen_films'].value_counts().reset_index()
            seen_counts.columns = ['seen_films', 'count']
            chart = alt.Chart(seen_counts).mark_bar().encode(
                x=alt.X('seen_films:N', title='Seen Films'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['seen_films', 'count']
            ).properties(title="Film Viewing Distribution", width=600, height=400)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.write("Column 'seen_films' not found.")
    # --- Demographics ---
    with vis_tabs[2]:
        st.subheader("Demographics Analysis")
        for col in ["Gender", "Age", "Household Income", "Education", "Location (Census Region)"]:
            if col in df_clean.columns:
                counts = df_clean[col].value_counts().reset_index()
                counts.columns = [col, "count"]
                chart = alt.Chart(counts).mark_bar().encode(
                    x=alt.X(f"{col}:N", title=col),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=[col, "count"]
                ).properties(title=f"{col} Distribution", width=300, height=300)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write(f"Column '{col}' not found.")
    # --- Film Ranking ---
    with vis_tabs[3]:
        st.subheader("Film Ranking Analysis")
        if 'film_ranking' in df_clean.columns:
            try:
                df_clean['film_ranking_numeric'] = pd.to_numeric(df_clean['film_ranking'], errors='coerce')
                ranking_chart = alt.Chart(df_clean).mark_bar().encode(
                    alt.X("film_ranking_numeric:Q", bin=alt.Bin(maxbins=10), title="Film Ranking"),
                    y=alt.Y('count()', title='Count'),
                    tooltip=['count()']
                ).properties(title="Film Ranking Histogram", width=600, height=400)
                st.altair_chart(ranking_chart, use_container_width=True)
            except Exception as e:
                st.write(f"Error: {e}")
        else:
            st.write("Column 'film_ranking' not found.")
    # --- Character Opinions ---
    with vis_tabs[4]:
        st.subheader("Character Opinions Analysis")
        if 'character_opinions' in df_clean.columns:
            opinions = df_clean['character_opinions'].value_counts().reset_index()
            opinions.columns = ['character_opinions', 'count']
            opinion_chart = alt.Chart(opinions).mark_bar().encode(
                x=alt.X('character_opinions:N', title='Character Opinion'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['character_opinions', 'count']
            ).properties(title="Character Opinions Distribution", width=600, height=400)
            st.altair_chart(opinion_chart, use_container_width=True)
        else:
            st.write("Column 'character_opinions' not found.")

# =========================
# TAB 3: Basic Statistical Reports
# =========================
with main_tabs[2]:
    st.header("Basic Statistical Reports")
    st.markdown("### Descriptive Statistics")
    st.dataframe(df_clean.describe(include='all'))
    
    st.markdown("### Histograms for Numeric Variables")
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(df_clean[col].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

# =========================
# TAB 4: Enhanced Dashboard & Export
# =========================
with main_tabs[3]:
    st.header("Enhanced Dashboard & Data Export")
    df_dashboard = df_clean.copy()
    if "Gender" in df_dashboard.columns:
        genders = sorted(df_dashboard["Gender"].unique().tolist())
        selected_genders = st.multiselect("Select Gender(s):", options=genders, default=genders)
        df_dashboard = df_dashboard[df_dashboard["Gender"].isin(selected_genders)]
    if "Age" in df_dashboard.columns:
        ages = sorted(df_dashboard["Age"].unique().tolist())
        selected_ages = st.multiselect("Select Age Group(s):", options=ages, default=ages)
        df_dashboard = df_dashboard[df_dashboard["Age"].isin(selected_ages)]
    st.markdown("#### Filtered Data")
    st.dataframe(df_dashboard)
    csv_data = df_dashboard.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

# =========================
# TAB 5: Geospatial Visualization
# =========================
with main_tabs[4]:
    st.header("Geospatial Visualization")
    if "Location (Census Region)" in df_clean.columns:
        # Dummy coordinates for known Census Regions
        region_coords = {
            "New England": {"lat": 42.0, "lon": -71.0},
            "Mid-Atlantic": {"lat": 40.0, "lon": -74.0},
            "East North Central": {"lat": 41.0, "lon": -87.0},
            "West North Central": {"lat": 39.0, "lon": -95.0},
            "South Atlantic": {"lat": 33.0, "lon": -80.0},
            "East South Central": {"lat": 32.0, "lon": -85.0},
            "West South Central": {"lat": 31.0, "lon": -100.0},
            "Mountain": {"lat": 39.0, "lon": -105.0},
            "Pacific": {"lat": 37.0, "lon": -120.0}
        }
        df_geo = df_clean.copy()
        df_geo = df_geo[df_geo["Location (Census Region)"].isin(region_coords.keys())]
        df_geo["lat"] = df_geo["Location (Census Region)"].apply(lambda x: region_coords[x]["lat"])
        df_geo["lon"] = df_geo["Location (Census Region)"].apply(lambda x: region_coords[x]["lon"])
        st.map(df_geo[["lat", "lon"]])
    else:
        st.write("No 'Location (Census Region)' column found.")

# =========================
# TAB 6: User Guide
# =========================
with main_tabs[5]:
    st.header("User Guide")
    st.markdown("""
    ### User Guide
    - **Data Cleaning:** Clean your survey data by dropping extraneous columns, renaming verbose columns, and handling missing values.
    - **Basic Visualization:** Explore initial charts on fan status, film viewing, demographics, film ranking, and character opinions.
    - **Basic Statistical Reports:** View descriptive statistics and histograms for numeric variables.
    - **Enhanced Dashboard & Export:** Apply filters and download the resulting data as CSV.
    - **Geospatial Visualization:** Visualize survey responses by mapping Census Regions.
    """)
    
st.markdown("#### Developed by Mr. Sangam S Bhamare 2025")
