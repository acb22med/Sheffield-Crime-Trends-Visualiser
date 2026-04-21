import pandas as pd, sqlite3

def clean(path="../data/raw.csv"):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "category": "crime_category",
        "location.street.name": "street_name",
        "location.latitude": "latitude",
        "location.longitude": "longitude",
        "outcome_status.category": "outcome_category",
    })
    
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    # df["latitude"] = df.to_numeric(df['latitude'], errors='coerce')
    # df["longitude"] = df.to_numeric(df['longitude'], errors='coerce')
    
    df = df[
        (df["latitude"].between(53.30, 53.47)) &
        (df["longitude"].between(-1.6, -1.30))
    ]
    
    df = df.dropna(subset=["latitude", "longitude", "month"])
    df = df.drop_duplicates()
    
    df["date"] = pd.to_datetime(df["month"])
    df["year"] = df["date"].dt.year
    df["month_n"] = df["date"].dt.month
    df["season"] = df["month_n"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn"
    })
    
    df.to_csv("../data/clean.csv", index=False)
    conn = sqlite3.connect("../data/crime.db")
    df.to_sql("crime", conn, if_exists="replace", index=False)
    conn.close()
    print(f"clean records: {len(df)}")
    print(f"Cleaned data saved to CSV and SQLite. Total records: {len(df)}")
    return df

if __name__ == "__main__":
    clean()