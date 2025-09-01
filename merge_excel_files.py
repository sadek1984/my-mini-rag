import pandas as pd
from pathlib import Path
import re
import sys
from datetime import datetime


# Add the 'src' directory to the Python path to allow importing our custom utility
sys.path.append('src')

try:
    # Import the robust Excel loading function from our data cleaning utility
    from src.utils.data_cleaning import load_and_prepare_excel
except ImportError:
    print("Error: Could not import 'load_and_prepare_excel'.")
    print("Please ensure this script is run from the root directory of your 'my-mini-rag' project,")
    print("and the file 'src/utils/data_cleaning.py' exists.")
    sys.exit(1)

# --- CONFIGURATION ---
# Define the main directory where the year's data is stored.
# Adjust this path if your data is in a different location.
DATA_DIRECTORY = Path("./src/data/2024")

# List of month folders to process.
MONTHS = ["Jan", "Feb", "March", "April", "May", "June",
          "July", "Aug", "Sep", "Oct", "Nov", "Dec"]

# --- SCRIPT LOGIC ---

def normalize_column_name(col_name: str) -> str:
    """
    Standardizes a column name by removing newlines, extra spaces,
    and converting to lowercase to ensure consistency across all files.
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)
    # Replace newlines and multiple spaces with a single space
    normalized = re.sub(r'\s+', ' ', col_name).strip().lower()
    return normalized

def merge_monthly_data():
    """
    Main function to find, load, and merge all Excel files from monthly folders.
    """
    if not DATA_DIRECTORY.is_dir():
        print(f"Error: Data directory not found at '{DATA_DIRECTORY}'")
        print("Please check the DATA_DIRECTORY path in the script.")
        return

    print("Starting the Excel file merging process...")
    all_dataframes = []
    total_files_processed = 0

    # Loop through each month's folder
    for month in MONTHS:
        month_path = DATA_DIRECTORY / month
        if not month_path.is_dir():
            print(f"--- Skipping month '{month}', directory not found.")
            continue

        print(f"--- Processing month: '{month}' ---")
        
        # Find all .xlsx files in the directory
        files_in_month = list(month_path.glob("*.xlsx"))
        if not files_in_month:
            print("No Excel files found in this month's folder.")
            continue

        # Loop through each Excel file found
        for file_path in files_in_month:
            print(f"Reading file: {file_path.name}...")
            
            # Use our robust function to load the data
            df = load_and_prepare_excel(str(file_path))
            
            if df is not None and not df.empty:
                # Normalize column names for consistent merging
                df.columns = [normalize_column_name(col) for col in df.columns]
                all_dataframes.append(df)
                total_files_processed += 1
            else:
                print(f"  -> Skipped {file_path.name} (no valid data found).")

    # Check if any data was collected
    if not all_dataframes:
        print("\nProcess finished, but no data was found to merge.")
        return

    print(f"\nMerging data from {total_files_processed} files...")
    
    # Concatenate all dataframes into a single master dataframe
    try:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
    except Exception as e:
        print(f"An error occurred during the final merge: {e}")
        return

    # Generate a unique output filename with the current date
    today_str = datetime.now().strftime("%Y-%m-%d")
    output_filename = f"merged_pesticide_data_{today_str}.xlsx"

    print(f"Saving merged data to '{output_filename}'...")
    
    # Save the final dataframe to a new Excel file
    merged_df.to_excel(output_filename, index=False)
    
    print("\n--- Success! ---")
    print(f"Total rows in merged file: {len(merged_df)}")
    print(f"Master file created at: {Path(output_filename).resolve()}")


if __name__ == "__main__":
    merge_monthly_data()
