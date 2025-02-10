import streamlit as st
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Star Wars Survey Data Cleaning", layout="wide")

st.title("Star Wars Survey Data Cleaning App")

# -------------------------------
# Helper: Load the data (cached)
# -------------------------------
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    # Assuming the CSV file is in the same directory as this script.
    df = pd.read_csv(filepath, sep="\t")  # adjust sep if necessary (e.g., comma-separated)
    return df

# Change the path to your data file
data_path = "star_wars.csv"
df_raw = load_data(data_path)

# Create a copy for cleaning
df_clean = df_raw.copy()

# Create tabs for different cleaning stages
tabs = st.tabs([
    "1. Original Data",
    "2. Drop Unwanted Columns",
    "3. Rename Columns",
    "4. Handle Missing Data",
    "5. Final Cleaned Data"
])

# -------------------------------
# Tab 1: Original Data
# -------------------------------
with tabs[0]:
    st.header("Original Data")
    st.write("Below are the first 10 rows of the raw data:")
    st.dataframe(df_raw.head(10))
    st.subheader("Data Summary")
    st.text(df_raw.info())

# -------------------------------
# Tab 2: Drop Unwanted Columns
# -------------------------------
with tabs[1]:
    st.header("Drop Unwanted Columns")
    # Identify columns that are not needed – here we drop columns that start with 'Unnamed'
    unnamed_cols = [col for col in df_clean.columns if col.startswith("Unnamed")]
    st.write("Columns to drop:", unnamed_cols)
    
    # Drop the unwanted columns
    df_clean = df_clean.drop(columns=unnamed_cols)
    
    st.write("Data after dropping unwanted columns (first 10 rows):")
    st.dataframe(df_clean.head(10))

# -------------------------------
# Tab 3: Rename Columns
# -------------------------------
with tabs[2]:
    st.header("Rename Columns")
    # Create a mapping dictionary to rename columns for clarity.
    # (Customize these mappings as needed based on your analysis.)
    rename_mapping = {
        "Have you seen any of the 6 films in the Star Wars franchise?": "seen_films",
        "Do you consider yourself to be a fan of the Star Wars film franchise?": "is_fan",
        "Which of the following Star Wars films have you seen? Please select all that apply.": "films_seen",
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "film_ranking",
        "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "character_opinions",
        # Add more mappings as needed for your analysis
    }
    st.write("Renaming columns as follows:")
    st.write(rename_mapping)
    
    # Rename the columns (only if they exist in the DataFrame)
    df_clean = df_clean.rename(columns={k: v for k, v in rename_mapping.items() if k in df_clean.columns})
    
    st.write("Data with renamed columns (first 10 rows):")
    st.dataframe(df_clean.head(10))

# -------------------------------
# Tab 4: Handle Missing Data
# -------------------------------
with tabs[3]:
    st.header("Handle Missing Data")
    st.subheader("Missing Data Counts")
    missing_counts = df_clean.isna().sum()
    st.dataframe(missing_counts.to_frame("Missing Count"))
    
    st.write("Handling missing values:")
    # Example strategies:
    # - For numeric columns, you might fill missing values with the median.
    # - For categorical columns, you might fill missing values with "Not Specified" or drop rows.
    # In this example, we fill missing values for all columns with a placeholder.
    
    # Determine data types for a simple fill strategy
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
        else:
            df_clean[col] = df_clean[col].fillna("Not Specified")
    
    st.write("Data after handling missing values (first 10 rows):")
    st.dataframe(df_clean.head(10))

# -------------------------------
# Tab 5: Final Cleaned Data
# -------------------------------
with tabs[4]:
    st.header("Final Cleaned Data Preview")
    st.write("Below is the preview of the final cleaned data:")
    st.dataframe(df_clean.head(10))
    
    st.subheader("DataFrame Info")
    buffer = []
    df_clean.info(buf=buffer)
    s = "\n".join(buffer)
    st.text(s)

# -------------------------------
# GitHub Repository
# -------------------------------
st.sidebar.header("Repository Info")
st.sidebar.markdown(
    """
    This code is hosted on [GitHub](https://github.com/yourusername/starwars-data-cleaning).
    
    **How to run locally:**
    1. Clone the repository.
    2. Install the requirements: `pip install streamlit pandas`.
    3. Run the app: `streamlit run starwars_data_cleaning_app.py`.
    """
)
