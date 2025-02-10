import streamlit as st
import pandas as pd

# Set the page configuration
st.set_page_config(page_title="Star Wars Survey Data Cleaning", layout="wide")

st.title("Star Wars Survey Data Cleaning App")

# Sidebar: GitHub and file uploader info
st.sidebar.header("Repository & File Upload")
st.sidebar.markdown(
    """
    This code is hosted on [GitHub](https://github.com/yourusername/starwars-data-cleaning).

    **How to run locally:**
    1. Clone the repository.
    2. Install the requirements: `pip install streamlit pandas`.
    3. Run the app: `streamlit run starwars_data_cleaning_app.py`.
    """
)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# -------------------------------
# Helper Function to Load Data
# -------------------------------
@st.cache_data
def load_data(file_source) -> pd.DataFrame:
    """
    Attempts to load a CSV file with tab delimiter first.
    If the resulting DataFrame has only one column,
    it retries with a comma delimiter.
    """
    try:
        # Try tab delimiter first
        df = pd.read_csv(file_source, delimiter="\t")
        # If all data is in one column, try a comma delimiter
        if df.shape[1] == 1:
            # Reset file pointer if file_source is a file-like object
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            df = pd.read_csv(file_source, delimiter=",")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    return df

# -------------------------------
# Load Data: from file uploader or default file
# -------------------------------
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    if df_raw is None:
        st.stop()
else:
    # If no file is uploaded, try to load the default file from disk
    try:
        df_raw = pd.read_csv("star_wars.csv", delimiter="\t")
        if df_raw.shape[1] == 1:
            df_raw = pd.read_csv("star_wars.csv", delimiter=",")
    except Exception as e:
        st.error("No file uploaded and the default file 'starwars_survey.csv' was not found or could not be loaded.")
        st.stop()

# Create a copy for cleaning operations
df_clean = df_raw.copy()

# -------------------------------
# Create Tabs for Cleaning Stages
# -------------------------------
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
    
    st.subheader("Data Information")
    # Capture DataFrame info in a string buffer
    buffer = []
    df_raw.info(buf=buffer)
    st.text("\n".join(buffer))

# -------------------------------
# Tab 2: Drop Unwanted Columns
# -------------------------------
with tabs[1]:
    st.header("Drop Unwanted Columns")
    # Identify columns that start with 'Unnamed' (often extra or blank columns)
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
    # Define a mapping dictionary to rename verbose column names to more manageable names.
    rename_mapping = {
        "Have you seen any of the 6 films in the Star Wars franchise?": "seen_films",
        "Do you consider yourself to be a fan of the Star Wars film franchise?": "is_fan",
        "Which of the following Star Wars films have you seen? Please select all that apply.": "films_seen",
        "Please rank the Star Wars films in order of preference with 1 being your favorite film in the franchise and 6 being your least favorite film.": "film_ranking",
        "Please state whether you view the following characters favorably, unfavorably, or are unfamiliar with him/her.": "character_opinions",
        # Add additional mappings here as needed.
    }
    st.write("Renaming columns using the following mapping:")
    st.write(rename_mapping)
    
    # Apply renaming only to columns that exist in the DataFrame
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
    # For numeric columns: fill missing values with the median.
    # For non-numeric columns: fill missing values with "Not Specified".
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
    
    st.subheader("Data Information")
    buffer = []
    df_clean.info(buf=buffer)
    st.text("\n".join(buffer))
