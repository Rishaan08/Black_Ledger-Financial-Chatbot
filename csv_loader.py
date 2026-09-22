import os
import sqlite3
import pandas as pd

csv_folder = "Datasets/Global Financial Data"
output_folder = "Cleaned Financial Data"

os.makedirs(output_folder, exist_ok=True)
db_path = os.path.join(output_folder, "financial_data.db")
schema_path = os.path.join(output_folder, "schema_summary.txt")

problem_files = ["ETFs.csv", "MutualFunds.csv", "INFRATEL.csv"]
no_date_files = [
    "2014_Financial_Data.csv",
    "2015_Financial_Data.csv",
    "2016_Financial_Data.csv",
    "2017_Financial_Data.csv",
    "2018_Financial_Data.csv",
]

def find_date_column(df):
    """Find whichever column looks like date column."""
    
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    
    return None

def get_date_range(df, date_col):
    """Try to get min and max from date column. Return None if it fails."""
    
    try:
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        min_date = dates.min()
        max_date = dates.max()
        if pd.isnull(min_date) or pd.isnull(max_date):
            return None, None
        return str(min_date.date()), str(max_date.date())
    
    except Exception:
        return None, None
    
def fix_duplicate_columns(df):
    """If a DataFrame has duplicate column names, rename them by adding _1, _2 etc."""
    seen = {}
    new_columns = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    df.columns = new_columns
    return df
    
def load_csv_to_sqlite():
    
    all_csv_files = []
    for root, dirs,files in os.walk(csv_folder):
        for file in files:
            if file.endswith(".csv"):
                all_csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_csv_files)} CSV files in {csv_folder}.")
    
    # Connect to SQLite
    conn = sqlite3.connect(db_path)
    
    schema_lines = []
    schema_lines.append("Financial Database Schema Summary")
    schema_lines.append("Data cutoff: Maximum 2021. Cannot answer questions after 2021.")
    
    loaded = 0
    skipped = 0
    
    for file_path in sorted(all_csv_files):
        filename = os.path.basename(file_path)
        table_name = filename.replace(".csv", "").replace(" ", "_").replace("-", "_")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.lower()
            df = fix_duplicate_columns(df)
            
            if df.empty:
                print(f"Skipped (empty): {filename}")
                skipped += 1
                continue
            
            # Store in SQLite
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            loaded += 1
            print(f"Loaded: {filename} -> table: {table_name} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Build schema summary entry
            schema_lines.append(f"\nTable: {table_name}")
            schema_lines.append(f"Source file: {filename}")
            schema_lines.append(f"Columns: {', '.join(df.columns.tolist())}")
            schema_lines.append(f"Total Rows: {len(df)}")
            
            if filename in problem_files:
                schema_lines.append("Date Range: Could not parse (timezone or format issue)")
                
            elif filename in no_date_files:
                schema_lines.append("Date Range: No date column - this is annual financial snapshot data (not time series)")
                
            else:
                date_col = find_date_column(df)
                if date_col:
                    min_d, max_d = get_date_range(df, date_col)
                    if min_d and max_d:
                        schema_lines.append(f"Date Range: {min_d} to {max_d}")
                    else:
                        schema_lines.append("Date Range: Could not extract")
                else:
                    schema_lines.append("Date Range: No date column found")
                    
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            skipped += 1
            
    conn.close()
    
    with open(schema_path, "w") as f:
        f.write("\n".join(schema_lines))
        
    print(f"Done. Loaded {loaded} tables into {db_path}")
    print(f"Schema summary written to {schema_path}")
    print(f"Skipped {skipped} files")
    
if __name__ == "__main__":
    load_csv_to_sqlite()