import streamlit as st
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Star Wars Survey Data Cleaning", layout="wide")

st.title("Star Wars Survey Data Cleaning App")

# -----------------------------------------------------------------------------
# Helper Function: Create a DataFrame with Data Information in Tabular Format
# -----------------------------------------------------------------------------
def get_df_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing information about the input DataFrame.
    Includes: Column name, Non-Null Count, Null Count, % Missing, Data Type,
    and Memory Usage (in Bytes).
    """
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values,
        "% Missing": (df.isnull().sum().values / len(df)) * 100,
        "Data Type": df.dtypes.astype(str).values
    })
    # Get memory usage for each column (excluding the index)
    mem_usage = df.memory_usage(deep=True)[df.columns]
    info_df["Memory Usage (Bytes)"] = mem_usage.values
    return info_df

# -----------------------------------------------------------------------------
# Helper Function: Load CSV File
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """
    Attempts to load a CSV file using a tab delimiter first.
    If the resulting DataFrame has only one column, it retries using a comma delimiter.
    """
    try:
        # Try reading with tab as delimiter
        df = pd.read_csv(filepath, delimiter="\t")
        # If data is still in one column, try using comma as delimiter
        if df.shape[1] == 1:
            df = pd.read_csv(filepath, delimiter=",")
    except Exception as e:
        st.error(f"Error reading file '{filepath}': {e}")
        return None
    return df

# -----------------------------------------------------------------------------
# Load Data from the fixed file "star_wars.csv"
# -----------------------------------------------------------------------------
data_path = "star_wars.csv"
df_raw = load_data(data_path)
if df_raw is None:
    st.stop()

# Create a copy for cleaning operations
df_clean = df_raw.copy()

# -----------------------------------------------------------------------------
# Create Tabs for Different Cleaning Stages
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "1. Original Data",
    "2. Drop Unwanted Columns",
    "3. Rename Columns",
    "4. Handle Missing Data",
    "5. Final Cleaned Data"
])

# -----------------------------------------------------------------------------
# Tab 1: Original Data
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("Original Data")
    st.write("Below are the first 10 rows of the raw data:")
    st.dataframe(df_raw.head(10))
    
    st.subheader("Data Information")
    st.dataframe(get_df_info(df_raw))

# -----------------------------------------------------------------------------
# Tab 2: Drop Unwanted Columns
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Drop Unwanted Columns")
    # Identify columns that start with 'Unnamed' (often extra or blank columns)
    unnamed_cols = [col for col in df_clean.columns if col.startswith("Unnamed")]
    st.write("Columns to drop:", unnamed_cols)
    
    # Drop the unwanted columns
    df_clean = df_clean.drop(columns=unnamed_cols)
    
    st.write("Data after dropping unwanted columns (first 10 rows):")
    st.dataframe(df_clean.head(10))

# -----------------------------------------------------------------------------
# Tab 3: Rename Columns
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Rename Columns")
    # Define a mapping dictionary to rename verbose column names to simpler names.
    rename_mapping = {
        "Have you seen any of the 6 films in the Star Wars franchise?": "seen_films",
        "Do you consider yourself to be a fan of the Star Wars film franchise?": "is_fan",
        "Which of the following Star Wars films have you seen? Please select all that apply.": "films_seen",
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "film_ranking",
        "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "character_opinions",
        # Add additional mappings as needed.
    }
    st.write("Renaming columns using the following mapping:")
    st.write(rename_mapping)
    
    # Rename the columns (only applying mapping for columns that exist)
    df_clean = df_clean.rename(columns={k: v for k, v in rename_mapping.items() if k in df_clean.columns})
    
    st.write("Data with renamed columns (first 10 rows):")
    st.dataframe(df_clean.head(10))

# -----------------------------------------------------------------------------
# Tab 4: Handle Missing Data
# -----------------------------------------------------------------------------
with tabs[3]:
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

# -----------------------------------------------------------------------------
# Tab 5: Final Cleaned Data
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Final Cleaned Data Preview")
    st.write("Below is the preview of the final cleaned data:")
    st.dataframe(df_clean.head(10))
    
    st.subheader("Data Information")
    st.dataframe(get_df_info(df_clean))
