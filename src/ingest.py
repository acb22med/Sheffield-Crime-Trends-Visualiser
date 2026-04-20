import requests, pandas as pd, sqlite3, time

# Step 1 — check what dates are actually available
def get_available_dates():
    r = requests.get("https://data.police.uk/api/crimes-street-dates")
    dates = r.json()
    return [d["date"] for d in dates]


SHEFFIELD_POLY = "53.4200,-1.5800:53.4200,-1.3500:53.3200,-1.3500:53.3200,-1.5800"

def fetch_month(date):
    url = "https://data.police.uk/api/crimes-street/all-crime"
    # use poly instead of lat/lng to control the area size
    params = {
        "poly": SHEFFIELD_POLY,
        "date": date
    }
    r = requests.get(url, params=params, timeout=30)
    
    print(f"  {date} — status: {r.status_code}")
    
    if r.status_code == 503:
        print("  503: too many crimes in area — splitting into quadrants")
        return fetch_month_quadrants(date)
    
    if r.status_code != 200:
        print(f"  Skipping {date} — unexpected status {r.status_code}")
        return pd.DataFrame()
    
    data = r.json()
    if not data:
        return pd.DataFrame()
    
    df = pd.json_normalize(data)
    df["fetched_month"] = date
    return df

def fetch_month_quadrants(date):
    """Split Sheffield into 4 quadrants if full area exceeds 10k crimes"""
    quadrants = [
        "53.4200,-1.5800:53.4200,-1.4650:53.3700,-1.4650:53.3700,-1.5800",
        "53.4200,-1.4650:53.4200,-1.3500:53.3700,-1.3500:53.3700,-1.4650",
        "53.3700,-1.5800:53.3700,-1.4650:53.3200,-1.4650:53.3200,-1.5800",
        "53.3700,-1.4650:53.3700,-1.3500:53.3200,-1.3500:53.3200,-1.4650",
    ]
    frames = []
    for poly in quadrants:
        r = requests.get(
            "https://data.police.uk/api/crimes-street/all-crime",
            params={"poly": poly, "date": date},
            timeout=30
        )
        if r.status_code == 200 and r.json():
            df = pd.json_normalize(r.json())
            df["fetched_month"] = date
            frames.append(df)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

if __name__ == "__main__":
    print("Checking available dates...")
    available = get_available_dates()
   
    months = available[:18]
    print(f"Fetching {len(months)} months: {months[-1]} to {months[0]}")
    
    frames = []
    for m in months:
        df = fetch_month(m)
        if not df.empty:
            frames.append(df)
        time.sleep(0.8)  
    
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv("../data/raw.csv", index=False)
        print(f"\nDone. Total records: {len(combined)}")
        print(combined.columns.tolist())  
    else:
        print("No data fetched — check your connection")