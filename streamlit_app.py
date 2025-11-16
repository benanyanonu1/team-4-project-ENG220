import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt

# -----------------------------------------------------------------------------
# Page configuration

st.set_page_config(
    page_title="ENG220 Team 4 Final Project – Violence & Security",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Data loading

@st.cache_data
def load_gun_violence_data():
    """
    Load gun violence data.

    Preferred: load from an online URL (e.g. Google Drive direct-download link).
    Fallback: local CSV in data/gun-violence-data_01-2013_03-2018.csv
    """

    # 👉 TODO: put your real direct CSV URL here once you host it (e.g. Google Drive)
    DATA_URL = ""  # e.g. "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"

    df = None

    # Try URL first if you’ve set one
    if DATA_URL:
        try:
            df = pd.read_csv(DATA_URL, parse_dates=["date"])
        except Exception as e:
            st.warning(f"Tried to load data from URL but failed: {e}. Falling back to local file. ({e})")

    if df is None:
        # Local fallback
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


@st.cache_data
def build_participant_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand participant_* columns into one row per participant.

    Columns: incident_id, state, year, age, age_group, gender,
             participant_type, status, relationship
    """
    required_cols = [
        "incident_id",
        "state",
        "year",
        "participant_age",
        "participant_age_group",
        "participant_gender",
        "participant_type",
        "participant_status",
        "participant_relationship",
    ]
    for c in required_cols:
        if c not in df.columns:
            return pd.DataFrame(columns=["incident_id", "state", "year"])

    rows = []
    cols_to_parse = [
        "participant_age",
        "participant_age_group",
        "participant_gender",
        "participant_type",
        "participant_status",
        "participant_relationship",
    ]

    for _, row in df[["incident_id", "state", "year"] + cols_to_parse].iterrows():
        fields = {}

        for col in cols_to_parse:
            val = row[col]
            if isinstance(val, str):
                parts = [p for p in val.split("||") if p]
                for p in parts:
                    if "::" in p:
                        idx, v = p.split("::", 1)
                        fields.setdefault(col, {})[idx] = v

        if not fields:
            continue

        indices = set()
        for mapping in fields.values():
            indices.update(mapping.keys())

        for idx in indices:
            age_val = fields.get("participant_age", {}).get(idx)
            try:
                age = int(age_val)
            except (TypeError, ValueError):
                age = None

            rows.append(
                {
                    "incident_id": row["incident_id"],
                    "state": row["state"],
                    "year": row["year"],
                    "age": age,
                    "age_group": fields.get("participant_age_group", {}).get(idx),
                    "gender": fields.get("participant_gender", {}).get(idx),
                    "participant_type": fields.get("participant_type", {}).get(idx),
                    "status": fields.get("participant_status", {}).get(idx),
                    "relationship": fields.get("participant_relationship", {}).get(idx),
                }
            )

    return pd.DataFrame(rows)


@st.cache_data
def build_gun_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand gun_* columns into one row per gun.

    Columns: incident_id, state, year, gun_type, gun_stolen
    """
    for c in ["incident_id", "state", "year", "gun_type", "gun_stolen"]:
        if c not in df.columns:
            return pd.DataFrame(columns=["incident_id", "state", "year", "gun_type", "gun_stolen"])

    rows = []

    for _, row in df[["incident_id", "state", "year", "gun_type", "gun_stolen"]].iterrows():
        gun_types = {}
        gun_stolen = {}

        if isinstance(row["gun_type"], str):
            for p in row["gun_type"].split("||"):
                if "::" in p:
                    idx, v = p.split("::", 1)
                    gun_types[idx] = v

        if isinstance(row["gun_stolen"], str):
            for p in row["gun_stolen"].split("||"):
                if "::" in p:
                    idx, v = p.split("::", 1)
                    gun_stolen[idx] = v

        if not gun_types and not gun_stolen:
            continue

        indices = set(gun_types.keys()) | set(gun_stolen.keys())

        for idx in indices:
            rows.append(
                {
                    "incident_id": row["incident_id"],
                    "state": row["state"],
                    "year": row["year"],
                    "gun_type": gun_types.get(idx),
                    "gun_stolen": gun_stolen.get(idx),
                }
            )

    return pd.DataFrame(rows)


# Load base data
data = load_gun_violence_data()
participants_all = build_participant_table(data)
guns_all = build_gun_table(data)

# -----------------------------------------------------------------------------
# Title

st.title("ENG220 Team 4 Final Project")
st.subheader("Violence & Security")

st.divider()

# -----------------------------------------------------------------------------
# Sidebar filters (time + state)

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

# Base incident filter by year + state
base_incidents = data[
    (data["year"] >= from_year)
    & (data["year"] <= to_year)
    & (data["state"].isin(selected_states))
]

if base_incidents.empty:
    st.warning("No incidents match the current year/state filters.")
    st.stop()

base_incident_ids = set(base_incidents["incident_id"].unique())

# -----------------------------------------------------------------------------
# Sidebar filters (participant + gun)

st.sidebar.markdown("---")
st.sidebar.subheader("Participant filters (optional)")

# participant data limited to filtered incidents
p_base = participants_all[participants_all["incident_id"].isin(base_incident_ids)]

role_options = sorted(p_base["participant_type"].dropna().unique()) if not p_base.empty else []
gender_options = sorted(p_base["gender"].dropna().unique()) if not p_base.empty else []
rel_options = sorted(p_base["relationship"].dropna().unique()) if not p_base.empty else []

selected_roles = st.sidebar.multiselect(
    "Participant role (e.g. Victim, Subject-Suspect)",
    options=role_options,
)

selected_genders = st.sidebar.multiselect(
    "Participant gender",
    options=gender_options,
)

selected_relationships = st.sidebar.multiselect(
    "Relationship (e.g. Family, Partner)",
    options=rel_options,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Gun filters (optional)")

g_base = guns_all[guns_all["incident_id"].isin(base_incident_ids)]

gun_type_options = sorted(g_base["gun_type"].dropna().unique()) if not g_base.empty else []
stolen_options = sorted(g_base["gun_stolen"].dropna().unique()) if not g_base.empty else []

selected_gun_types = st.sidebar.multiselect(
    "Gun type (e.g. Handgun, Rifle)",
    options=gun_type_options,
)

selected_stolen = st.sidebar.multiselect(
    "Gun stolen status",
    options=stolen_options,
)

# -----------------------------------------------------------------------------
# Apply advanced filters to incidents

incident_ids = set(base_incident_ids)

# Participant-based incident filter
if p_base is not None and not p_base.empty:
    pf = p_base.copy()
    if selected_roles:
        pf = pf[pf["participant_type"].isin(selected_roles)]
    if selected_genders:
        pf = pf[pf["gender"].isin(selected_genders)]
    if selected_relationships:
        pf = pf[pf["relationship"].isin(selected_relationships)]

    if selected_roles or selected_genders or selected_relationships:
        incident_ids &= set(pf["incident_id"].unique())

# Gun-based incident filter
if g_base is not None and not g_base.empty:
    gf = g_base.copy()
    if selected_gun_types:
        gf = gf[gf["gun_type"].isin(selected_gun_types)]
    if selected_stolen:
        gf = gf[gf["gun_stolen"].isin(selected_stolen)]

    if selected_gun_types or selected_stolen:
        incident_ids &= set(gf["incident_id"].unique())

# Final incident filter
filtered = base_incidents[base_incidents["incident_id"].isin(incident_ids)]

if filtered.empty:
    st.warning("No incidents match all the selected filters.")
    st.stop()

# Also filter participant / gun tables for later charts
p_filtered = p_base[p_base["incident_id"].isin(incident_ids)]
guns_filtered = g_base[g_base["incident_id"].isin(incident_ids)]

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

st.divider()

# -----------------------------------------------------------------------------
# Participant demographics (age, gender, role, relationship)

st.subheader("Participant demographics")

if p_filtered.empty:
    st.info("No participant-level data available for the current filters.")
else:
    # Age distribution for victims vs suspects
    p_age = p_filtered.dropna(subset=["age"]).copy()
    p_age = p_age[(p_age["age"] >= 0) & (p_age["age"] <= 100)]

    col_age, col_gender = st.columns(2)

    with col_age:
        st.markdown("**Age distribution (victims vs suspects)**")
        if p_age.empty:
            st.info("No usable age data for the current filters.")
        else:
            age_chart = (
                alt.Chart(p_age)
                .mark_bar()
                .encode(
                    x=alt.X("age:Q", bin=alt.Bin(maxbins=30), title="Age"),
                    y=alt.Y("count():Q", title="Number of participants"),
                    color=alt.Color("participant_type:N", title="Type"),
                    tooltip=[
                        "participant_type:N",
                        "count()",
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(age_chart, use_container_width=True)

    with col_gender:
        st.markdown("**Gender by participant role**")
        g = p_filtered.dropna(subset=["gender", "participant_type"])
        if g.empty:
            st.info("No usable gender data for the current filters.")
        else:
            gender_counts = (
                g.groupby(["participant_type", "gender"], as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )

            gender_chart = (
                alt.Chart(gender_counts)
                .mark_bar()
                .encode(
                    x=alt.X("gender:N", title="Gender"),
                    y=alt.Y("count:Q", title="Number of participants"),
                    color=alt.Color("participant_type:N", title="Role"),
                    column=alt.Column("participant_type:N", title=""),
                    tooltip=["participant_type", "gender", "count"],
                )
                .properties(height=300)
            )
            st.altair_chart(gender_chart, use_container_width=True)

    # Relationship chart
    rel = p_filtered.dropna(subset=["relationship"])
    if not rel.empty:
        st.markdown("**Relationship to other participants (e.g. family, partner)**")
        rel_counts = (
            rel.groupby("relationship", as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .head(15)
        )
        rel_chart = (
            alt.Chart(rel_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Number of participants"),
                y=alt.Y("relationship:N", sort="-x", title="Relationship"),
                tooltip=["relationship", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(rel_chart, use_container_width=True)

st.divider()

# -----------------------------------------------------------------------------
# Gun characteristics (type + stolen + means/medians)

st.subheader("Gun characteristics")

if guns_filtered.empty:
    st.info("No gun-level data available for the current filters.")
else:
    guns_clean = guns_filtered.copy()
    guns_clean["gun_type"] = guns_clean["gun_type"].fillna("Unknown")
    guns_clean["gun_stolen"] = guns_clean["gun_stolen"].fillna("Unknown")

    gun_counts = (
        guns_clean.groupby("gun_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(10)
    )

    stolen_counts = (
        guns_clean.groupby("gun_stolen", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )

    col_gun_type, col_stolen = st.columns(2)

    with col_gun_type:
        st.markdown("**Top gun types**")
        gun_type_chart = (
            alt.Chart(gun_counts)
            .mark_bar()
            .encode(
                x=alt.X("count:Q", title="Number of guns"),
                y=alt.Y("gun_type:N", sort="-x", title="Gun type"),
                tooltip=["gun_type", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(gun_type_chart, use_container_width=True)

    with col_stolen:
        st.markdown("**Guns stolen or not**")
        stolen_chart = (
            alt.Chart(stolen_counts)
            .mark_bar()
            .encode(
                x=alt.X("gun_stolen:N", title="Stolen status"),
                y=alt.Y("count:Q", title="Number of guns"),
                tooltip=["gun_stolen", "count"],
            )
            .properties(height=300)
        )
        st.altair_chart(stolen_chart, use_container_width=True)

    # Mean / median killed per incident by gun type
    st.markdown("**Outcomes by gun type (mean & median deaths per incident)**")

    inc_level = filtered[["incident_id", "n_killed", "n_injured"]].drop_duplicates()
    inc_gun = guns_clean[["incident_id", "gun_type"]].dropna().drop_duplicates()
    merged = inc_gun.merge(inc_level, on="incident_id", how="left")

    gun_stats = (
        merged.groupby("gun_type", as_index=False)
        .agg(
            incidents=("incident_id", "nunique"),
            mean_killed=("n_killed", "mean"),
            median_killed=("n_killed", "median"),
            mean_total_victims=("n_killed", "mean"),  # you can change this if you add injured
        )
        .sort_values("mean_killed", ascending=False)
    )

    # Round for display
    gun_stats["mean_killed"] = gun_stats["mean_killed"].round(2)
    gun_stats["median_killed"] = gun_stats["median_killed"].round(2)

    st.dataframe(
        gun_stats.head(15),
        use_container_width=True,
        height=350,
    )

st.divider()

# -----------------------------------------------------------------------------
# Map of incidents (hoverable)

st.subheader("Map of incidents (hover for details)")

if "latitude" in filtered.columns and "longitude" in filtered.columns:
    map_source = filtered[
        [
            "latitude",
            "longitude",
            "date",
            "state",
            "city_or_county",
            "n_killed",
            "n_injured",
            "incident_characteristics",
        ]
    ].dropna(subset=["latitude", "longitude"])

    if len(map_source) == 0:
        st.info(
            "There are no incidents with latitude/longitude in the current filters, "
            "so a map cannot be drawn."
        )
    else:
        # Sample if huge
        if len(map_source) > 5000:
            map_source = map_source.sample(n=5000, random_state=0)

        map_chart = (
            alt.Chart(map_source)
            .mark_circle(opacity=0.6)
            .encode(
                longitude="longitude:Q",
                latitude="latitude:Q",
                size=alt.Size("n_killed:Q", title="People killed", scale=alt.Scale(range=[10, 200])),
                tooltip=[
                    "date:T",
                    "state:N",
                    "city_or_county:N",
                    "n_killed:Q",
                    "n_injured:Q",
                    "incident_characteristics:N",
                ],
            )
            .properties(height=450)
        )
        st.altair_chart(map_chart, use_container_width=True)
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