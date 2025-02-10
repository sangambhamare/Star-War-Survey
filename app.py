import streamlit as st
import pandas as pd
import altair as alt

# Set the page configuration
st.set_page_config(page_title="Star Wars Survey Data Cleaning & Visualization", layout="wide")

st.title("Star Wars Survey Data Cleaning & Visualization App")

# =============================================================================
# Helper Function: Create a DataFrame with Data Information in Tabular Format
# =============================================================================
def get_df_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame summarizing the input DataFrame's information:
    Column, Non-Null Count, Null Count, % Missing, Data Type, and Memory Usage.
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

# =============================================================================
# Helper Function: Load CSV File
# =============================================================================
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """
    Attempts to load a CSV file using a tab delimiter first.
    If the resulting DataFrame has only one column, it retries using a comma delimiter.
    """
    try:
        df = pd.read_csv(filepath, delimiter="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, delimiter=",")
    except Exception as e:
        st.error(f"Error reading file '{filepath}': {e}")
        return None
    return df

# =============================================================================
# Load Data from the Fixed File "star_wars.csv"
# =============================================================================
data_path = "star_wars.csv"
df_raw = load_data(data_path)
if df_raw is None:
    st.stop()

# Create a copy for cleaning operations
df_clean = df_raw.copy()

# =============================================================================
# Main Tabs: Data Cleaning and Data Visualization
# =============================================================================
main_tabs = st.tabs(["Data Cleaning", "Data Visualization"])

# -----------------------------------------------------------------------------
# Data Cleaning Tab
# -----------------------------------------------------------------------------
with main_tabs[0]:
    # Create nested tabs for each cleaning stage
    cleaning_tabs = st.tabs([
        "1. Original Data",
        "2. Drop Unwanted Columns",
        "3. Rename Columns",
        "4. Handle Missing Data",
        "5. Final Cleaned Data"
    ])

    # --- Tab 1: Original Data ---
    with cleaning_tabs[0]:
        st.header("Original Data")
        st.write("Below are the first 10 rows of the raw data:")
        st.dataframe(df_raw.head(10))
        
        st.subheader("Data Information")
        st.dataframe(get_df_info(df_raw))

    # --- Tab 2: Drop Unwanted Columns ---
    with cleaning_tabs[1]:
        st.header("Drop Unwanted Columns")
        # Identify columns that start with 'Unnamed' (often extra or blank columns)
        unnamed_cols = [col for col in df_clean.columns if col.startswith("Unnamed")]
        st.write("Columns to drop:", unnamed_cols)
        df_clean = df_clean.drop(columns=unnamed_cols)
        st.write("Data after dropping unwanted columns (first 10 rows):")
        st.dataframe(df_clean.head(10))

    # --- Tab 3: Rename Columns ---
    with cleaning_tabs[2]:
        st.header("Rename Columns")
        # Define mapping to rename verbose column names to simpler ones
        rename_mapping = {
            "Have you seen any of the 6 films in the Star Wars franchise?": "seen_films",
            "Do you consider yourself to be a fan of the Star Wars film franchise?": "is_fan",
            "Which of the following Star Wars films have you seen? Please select all that apply.": "films_seen",
            "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "film_ranking",
            "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "character_opinions",
        }
        st.write("Renaming columns using the following mapping:")
        st.write(rename_mapping)
        df_clean = df_clean.rename(columns={k: v for k, v in rename_mapping.items() if k in df_clean.columns})
        st.write("Data with renamed columns (first 10 rows):")
        st.dataframe(df_clean.head(10))

    # --- Tab 4: Handle Missing Data ---
    with cleaning_tabs[3]:
        st.header("Handle Missing Data")
        st.subheader("Missing Data Counts")
        missing_counts = df_clean.isna().sum()
        st.dataframe(missing_counts.to_frame("Missing Count"))
        
        st.write("Handling missing values:")
        # For numeric columns, fill missing values with the median.
        # For non-numeric columns, fill missing values with "Not Specified".
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
            else:
                df_clean[col] = df_clean[col].fillna("Not Specified")
        st.write("Data after handling missing values (first 10 rows):")
        st.dataframe(df_clean.head(10))

    # --- Tab 5: Final Cleaned Data ---
    with cleaning_tabs[4]:
        st.header("Final Cleaned Data Preview")
        st.write("Below is the preview of the final cleaned data:")
        st.dataframe(df_clean.head(10))
        
        st.subheader("Data Information")
        st.dataframe(get_df_info(df_clean))

# -----------------------------------------------------------------------------
# Data Visualization Tab
# -----------------------------------------------------------------------------
with main_tabs[1]:
    st.header("Data Visualization")
    # Create nested tabs for various types of plots
    vis_tabs = st.tabs(["Fan Analysis", "Film Viewing", "Demographics", "Film Ranking", "Character Opinions"])

    # --- Sub-tab: Fan Analysis ---
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
            st.write("Column 'is_fan' not found in the data.")

    # --- Sub-tab: Film Viewing ---
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
            st.write("Column 'seen_films' not found in the data.")

    # --- Sub-tab: Demographics ---
    with vis_tabs[2]:
        st.subheader("Demographics Analysis")
        # Gender Distribution
        if 'Gender' in df_clean.columns:
            gender_counts = df_clean['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'count']
            gender_chart = alt.Chart(gender_counts).mark_bar().encode(
                x=alt.X('Gender:N', title='Gender'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['Gender', 'count']
            ).properties(title="Gender Distribution", width=300, height=300)
            st.altair_chart(gender_chart, use_container_width=True)
        else:
            st.write("Column 'Gender' not found.")
        
        # Age Distribution
        if 'Age' in df_clean.columns:
            age_counts = df_clean['Age'].value_counts().reset_index()
            age_counts.columns = ['Age', 'count']
            age_chart = alt.Chart(age_counts).mark_bar().encode(
                x=alt.X('Age:N', title='Age'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['Age', 'count']
            ).properties(title="Age Distribution", width=300, height=300)
            st.altair_chart(age_chart, use_container_width=True)
        else:
            st.write("Column 'Age' not found.")
        
        # Household Income Distribution
        if 'Household Income' in df_clean.columns:
            income_counts = df_clean['Household Income'].value_counts().reset_index()
            income_counts.columns = ['Household Income', 'count']
            income_chart = alt.Chart(income_counts).mark_bar().encode(
                x=alt.X('Household Income:N', title='Household Income'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['Household Income', 'count']
            ).properties(title="Household Income Distribution", width=300, height=300)
            st.altair_chart(income_chart, use_container_width=True)
        else:
            st.write("Column 'Household Income' not found.")
        
        # Education Distribution
        if 'Education' in df_clean.columns:
            education_counts = df_clean['Education'].value_counts().reset_index()
            education_counts.columns = ['Education', 'count']
            education_chart = alt.Chart(education_counts).mark_bar().encode(
                x=alt.X('Education:N', title='Education'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['Education', 'count']
            ).properties(title="Education Distribution", width=300, height=300)
            st.altair_chart(education_chart, use_container_width=True)
        else:
            st.write("Column 'Education' not found.")
        
        # Location (Census Region) Distribution
        if 'Location (Census Region)' in df_clean.columns:
            location_counts = df_clean['Location (Census Region)'].value_counts().reset_index()
            location_counts.columns = ['Location (Census Region)', 'count']
            location_chart = alt.Chart(location_counts).mark_bar().encode(
                x=alt.X('Location (Census Region):N', title='Location (Census Region)'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['Location (Census Region)', 'count']
            ).properties(title="Location Distribution", width=300, height=300)
            st.altair_chart(location_chart, use_container_width=True)
        else:
            st.write("Column 'Location (Census Region)' not found.")

    # --- Sub-tab: Film Ranking ---
    with vis_tabs[3]:
        st.subheader("Film Ranking Analysis")
        if 'film_ranking' in df_clean.columns:
            # Attempt to convert film_ranking to numeric
            try:
                df_clean['film_ranking_numeric'] = pd.to_numeric(df_clean['film_ranking'], errors='coerce')
                ranking_chart = alt.Chart(df_clean).mark_bar().encode(
                    alt.X("film_ranking_numeric:Q", bin=alt.Bin(maxbins=10), title="Film Ranking"),
                    y=alt.Y('count()', title='Count'),
                    tooltip=['count()']
                ).properties(title="Film Ranking Histogram", width=600, height=400)
                st.altair_chart(ranking_chart, use_container_width=True)
            except Exception as e:
                st.write(f"Error converting film_ranking to numeric: {e}")
        else:
            st.write("Column 'film_ranking' not found in the data.")

    # --- Sub-tab: Character Opinions ---
    with vis_tabs[4]:
        st.subheader("Character Opinions Analysis")
        if 'character_opinions' in df_clean.columns:
            opinion_counts = df_clean['character_opinions'].value_counts().reset_index()
            opinion_counts.columns = ['character_opinions', 'count']
            opinion_chart = alt.Chart(opinion_counts).mark_bar().encode(
                x=alt.X('character_opinions:N', title='Character Opinion'),
                y=alt.Y('count:Q', title='Count'),
                tooltip=['character_opinions', 'count']
            ).properties(title="Character Opinions Distribution", width=600, height=400)
            st.altair_chart(opinion_chart, use_container_width=True)
        else:
            st.write("Column 'character_opinions' not found in the data.")
