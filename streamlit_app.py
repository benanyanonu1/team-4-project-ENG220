import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt

# -----------------------------------------------------------------------------
# Page configuration

st.set_page_config(
    page_title="ENG 220 Gun Violence Project",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data loading

@st.cache_data
def load_gun_violence_data():
    """
    Load gun violence data.

    Preferred: load from an online URL (e.g. Google Drive direct-download link)
    so you don't have to keep the huge CSV inside the GitHub repo.

    Fallback: try to read from a local file in data/gun-violence-data_01-2013_03-2018.csv
    (this will work on your laptop, but probably not on Streamlit Cloud unless
    you manage the big file with LFS or similar).
    """

    # 👉 TODO: put your real direct CSV URL here once you host it (e.g. Google Drive)
    DATA_URL = ""  # e.g. "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

    df = None

    # Try URL first if you’ve set one
    if DATA_URL:
        try:
            df = pd.read_csv(DATA_URL, parse_dates=["date"])
        except Exception as e:
            st.warning(f"Tried to load data from URL but failed: {e}. Falling back to local file.")

    if df is None:
        # Local fallback (works when you have the CSV in the repo / on your machine)
        local_path = Path(__file__).parent / "data" / "gun-violence-data_01-2013_03-2018.csv"
        df = pd.read_csv(local_path, parse_dates=["date"])

    # Basic time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Make sure numeric columns are numeric
    for col in ["n_killed", "n_injured"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    return df


# Load data once (cached by Streamlit)
data = load_gun_violence_data()

# -----------------------------------------------------------------------------
# Title / Intro

st.title("ENG 220 Gun Violence Project")
st.markdown(
    """
This dashboard uses incident-level data on gun violence in the United States
(2013–2018). Use the filters on the left to explore **where** and **when** incidents
occur, and how many people are **killed** or **injured**.

You can use the visuals here to support arguments about **patterns of gun violence**
and the kinds of **policy responses** that might help reduce it.
"""
)

st.divider()

# -----------------------------------------------------------------------------
# Sidebar filters

st.sidebar.header("Filters")

min_year = int(data["year"].min())
max_year = int(data["year"].max())

year_range = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

from_year, to_year = year_range

states = sorted(data["state"].dropna().unique())

# Some common high-incident states as defaults (if present)
default_states = [
    "Illinois",
    "California",
    "Florida",
    "Texas",
    "Ohio",
    "New York",
]
default_states = [s for s in default_states if s in states] or states[:6]

selected_states = st.sidebar.multiselect(
    "States to include",
    options=states,
    default=default_states,
)

if not selected_states:
    st.warning("Select at least one state in the sidebar.")
    st.stop()

# -----------------------------------------------------------------------------
# Filtered dataset

filtered = data[
    (data["year"] >= from_year)
    & (data["year"] <= to_year)
    & (data["state"].isin(selected_states))
]

if filtered.empty:
    st.warning("No incidents match your current filters.")
    st.stop()

# -----------------------------------------------------------------------------
# Summary metrics

total_incidents = len(filtered)
total_killed = int(filtered["n_killed"].sum())
total_injured = int(filtered["n_injured"].sum())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Incidents (filtered)", f"{total_incidents:,}")

with col2:
    st.metric("People killed (filtered)", f"{total_killed:,}")

with col3:
    st.metric("People injured (filtered)", f"{total_injured:,}")

st.caption("These metrics update automatically when you change the filters.")
st.divider()

# -----------------------------------------------------------------------------
# Incidents over time (monthly)

st.subheader("Incidents over time")

monthly = (
    filtered.groupby(["month", "state"], as_index=False)
    .agg(
        incidents=("date", "count"),
        n_killed=("n_killed", "sum"),
        n_injured=("n_injured", "sum"),
    )
    .sort_values("month")
)

time_metric = st.radio(
    "What do you want to visualize?",
    options=["incidents", "n_killed", "n_injured"],
    format_func=lambda x: {
        "incidents": "Number of incidents",
        "n_killed": "Number killed",
        "n_injured": "Number injured",
    }[x],
    horizontal=True,
)

y_title = {
    "incidents": "Number of incidents",
    "n_killed": "People killed",
    "n_injured": "People injured",
}[time_metric]

time_chart = (
    alt.Chart(monthly)
    .mark_line(point=True)
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y(f"{time_metric}:Q", title=y_title),
        color=alt.Color("state:N", title="State"),
        tooltip=[
            "month:T",
            "state:N",
            "incidents:Q",
            "n_killed:Q",
            "n_injured:Q",
        ],
    )
    .properties(height=400)
    .interactive()
)

st.altair_chart(time_chart, use_container_width=True)

st.markdown(
    """
**How to use this:**  
Look for trends after particular years or events (e.g. new laws, major shootings).
Increasing trends might suggest the need for stronger interventions, while flat or
declining trends can show where policies may be working.
"""
)

st.divider()

# -----------------------------------------------------------------------------
# State comparison (bar charts)

st.subheader("State comparison")

state_summary = (
    filtered.groupby("state", as_index=False)
    .agg(
        incidents=("date", "count"),
        n_killed=("n_killed", "sum"),
        n_injured=("n_injured", "sum"),
    )
    .sort_values("incidents", ascending=False)
)

left, right = st.columns(2)

with left:
    st.markdown("**Incidents by state**")
    incidents_bar = (
        alt.Chart(state_summary)
        .mark_bar()
        .encode(
            x=alt.X("incidents:Q", title="Incidents"),
            y=alt.Y("state:N", sort="-x", title="State"),
            tooltip=["state", "incidents", "n_killed", "n_injured"],
        )
        .properties(height=300)
    )
    st.altair_chart(incidents_bar, use_container_width=True)

with right:
    st.markdown("**People killed by state**")
    killed_bar = (
        alt.Chart(state_summary)
        .mark_bar()
        .encode(
            x=alt.X("n_killed:Q", title="People killed"),
            y=alt.Y("state:N", sort="-x", title="State"),
            tooltip=["state", "incidents", "n_killed", "n_injured"],
        )
        .properties(height=300)
    )
    st.altair_chart(killed_bar, use_container_width=True)

st.markdown(
    """
This comparison helps highlight which states are contributing most to the total
violence in your selected time range. In your paper/presentation, you might
connect this to differences in laws, urbanization, poverty, etc.
"""
)

st.divider()

# -----------------------------------------------------------------------------
# Map of incidents (if coordinates are available)

if "latitude" in filtered.columns and "longitude" in filtered.columns:
    st.subheader("Map of incidents (sample)")

    # Sample up to 5000 points so the map doesn't choke on huge data
    map_df = (
        filtered[["latitude", "longitude"]]
        .dropna()
        .sample(n=min(5000, len(filtered)), random_state=0)
    )

    st.map(map_df, use_container_width=True)

    st.caption(
        "This map shows a random sample of incidents (to keep the app responsive)."
    )
else:
    st.info(
        "This dataset does not appear to include latitude/longitude columns, "
        "so a map view is not available."
    )

st.divider()

# -----------------------------------------------------------------------------
# Raw data table

st.subheader("Raw data (filtered)")

show_cols = [
    col
    for col in [
        "date",
        "state",
        "city_or_county",
        "n_killed",
        "n_injured",
        "incident_characteristics",
    ]
    if col in filtered.columns
]

st.dataframe(
    filtered[show_cols].sort_values("date", ascending=False),
    use_container_width=True,
    height=400,
)

st.caption(
    "You can scroll, sort, and search within this table to find specific incidents "
    "that support your argument or case study."
)