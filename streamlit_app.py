import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import pydeck as pdk


st.set_page_config(
    page_title="ENG220 Team 4 Final Project Violence and Security",
    page_icon="ENG220",
    layout="wide",
)


@st.cache_data
def load_gun_violence_data():
    """Load gun violence data from url if provided, otherwise from local data folder."""

    data_url = ""  # optional url if you host the csv somewhere

    df = None

    if data_url:
        try:
            df = pd.read_csv(data_url, parse_dates=["date"])
        except Exception:
            st.warning("Could not load data from url, using local file instead")

    if df is None:
        local_path = Path(__file__).parent / "data" / "gun-violence-data_01-2013_03-2018.csv"
        df = pd.read_csv(local_path, parse_dates=["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    for col in ["n_killed", "n_injured"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0

    return df


@st.cache_data
def build_participant_table(df: pd.DataFrame) -> pd.DataFrame:
    """Turn participant columns into one row per participant."""

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

    p = pd.DataFrame(rows)

    if not p.empty and "participant_type" in p.columns:
        p["participant_type"] = p["participant_type"].replace({"Subject-Suspect": "Suspect"})

    if "gender" in p.columns:
        p["gender"] = p["gender"].astype("string").str.strip()
        p["gender"] = p["gender"].str.title()
        multi_mask = p["gender"].str.contains(",", na=False)
        p.loc[multi_mask, "gender"] = None

    return p


@st.cache_data
def build_gun_table(df: pd.DataFrame) -> pd.DataFrame:
    """Turn gun columns into one row per gun."""

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


def sidebar_multiselect_with_all(label: str, options, key_prefix: str):
    """Sidebar helper that gives an all toggle above a multiselect."""

    if not options:
        return []

    all_key = key_prefix + "_all"
    sel_key = key_prefix + "_sel"

    st.sidebar.markdown(f"**{label}**")
    all_selected = st.sidebar.checkbox(f"All {label.lower()}", value=True, key=all_key)

    if all_selected:
        st.sidebar.multiselect(
            "",
            options,
            default=options,
            key=sel_key,
            disabled=True,
        )
        return list(options)
    else:
        return st.sidebar.multiselect(
            "",
            options,
            key=sel_key,
        )


def main():
    try:
        data = load_gun_violence_data()
        participants_all = build_participant_table(data)
        guns_all = build_gun_table(data)

        st.title("ENG220 Team 4 Final Project")
        st.subheader(
            "Violence and Security — Ben Anyanonu, Max Aragon, Sebastian Gardiner, Avery Portillos, Ilan Raugust"
        )

        st.divider()

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

        selected_states = sidebar_multiselect_with_all("States to include", states, "states")

        if not selected_states:
            st.warning("Select at least one state in the sidebar.")
            st.stop()

        base_incidents = data[
            (data["year"] >= from_year)
            & (data["year"] <= to_year)
            & (data["state"].isin(selected_states))
        ]

        if base_incidents.empty:
            st.warning("No incidents match the current year and state filters.")
            st.stop()

        base_incident_ids = set(base_incidents["incident_id"].unique())

        st.sidebar.markdown("---")
        st.sidebar.subheader("Participant filters")

        p_base = participants_all[participants_all["incident_id"].isin(base_incident_ids)]

        role_options = sorted(p_base["participant_type"].dropna().unique()) if not p_base.empty else []
        gender_options = sorted(p_base["gender"].dropna().unique()) if not p_base.empty else []
        rel_options = sorted(p_base["relationship"].dropna().unique()) if not p_base.empty else []

        # start from all participants for this incident set
        if p_base is not None and not p_base.empty:
            pf = p_base.copy()
        else:
            pf = p_base

        selected_roles = sidebar_multiselect_with_all("Participant role", role_options, "roles")
        selected_genders = sidebar_multiselect_with_all("Participant gender", gender_options, "gender")
        selected_relationships = sidebar_multiselect_with_all(
            "Relationship for example family or partner", rel_options, "rel"
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("Gun filters")

        g_base = guns_all[guns_all["incident_id"].isin(base_incident_ids)]

        gun_type_options = sorted(g_base["gun_type"].dropna().unique()) if not g_base.empty else []
        stolen_options = sorted(g_base["gun_stolen"].dropna().unique()) if not g_base.empty else []

        selected_gun_types = sidebar_multiselect_with_all(
            "Gun type for example handgun or rifle", gun_type_options, "guntype"
        )
        selected_stolen = sidebar_multiselect_with_all("Gun stolen status", stolen_options, "stolen")

        incident_ids = set(base_incident_ids)

        # apply participant filters to participants table and incident set
        if pf is not None and not pf.empty:
            if selected_roles and len(selected_roles) != len(role_options):
                pf = pf[pf["participant_type"].isin(selected_roles)]
            if selected_genders and len(selected_genders) != len(gender_options):
                pf = pf[pf["gender"].isin(selected_genders)]
            if selected_relationships and len(selected_relationships) != len(rel_options):
                pf = pf[pf["relationship"].isin(selected_relationships)]

            if len(pf) < len(p_base):
                incident_ids &= set(pf["incident_id"].unique())

        # apply gun filters
        if g_base is not None and not g_base.empty:
            gf = g_base.copy()
            if selected_gun_types and len(selected_gun_types) != len(gun_type_options):
                gf = gf[gf["gun_type"].isin(selected_gun_types)]
            if selected_stolen and len(selected_stolen) != len(stolen_options):
                gf = gf[gf["gun_stolen"].isin(selected_stolen)]

            if len(gf) < len(g_base):
                incident_ids &= set(gf["incident_id"].unique())

        filtered = base_incidents[base_incidents["incident_id"].isin(incident_ids)]

        if filtered.empty:
            st.warning("No incidents match all selected filters.")
            st.stop()

        # important: participant demographics now respect participant filters, not just incident filters
        if pf is not None:
            p_filtered = pf[pf["incident_id"].isin(incident_ids)]
        else:
            p_filtered = pd.DataFrame(columns=p_base.columns if p_base is not None else [])

        guns_filtered = g_base[g_base["incident_id"].isin(incident_ids)]

        total_incidents = len(filtered)
        total_killed = int(filtered["n_killed"].sum())
        total_injured = int(filtered["n_injured"].sum())

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Incidents in view", f"{total_incidents:,}")

        with col2:
            st.metric("People killed in view", f"{total_killed:,}")

        with col3:
            st.metric("People injured in view", f"{total_injured:,}")

        st.divider()

        st.subheader("Incidents over time")

        monthly_state = (
            filtered.groupby(["month", "state"], as_index=False)
            .agg(
                n_incidents=("incident_id", "count"),
                n_killed=("n_killed", "sum"),
                n_injured=("n_injured", "sum"),
            )
            .sort_values("month")
        )

        monthly_state["month_label"] = monthly_state["month"].dt.strftime("%Y-%m")

        metric_key = st.radio(
            "What do you want to visualize",
            options=["incidents", "killed", "injured"],
            format_func=lambda x: {
                "incidents": "Number of incidents",
                "killed": "Number killed",
                "injured": "Number injured",
            }[x],
            horizontal=True,
        )

        metric_col_map = {
            "incidents": "n_incidents",
            "killed": "n_killed",
            "injured": "n_injured",
        }
        metric_label_map = {
            "incidents": "Number of incidents",
            "killed": "Number killed",
            "injured": "Number injured",
        }

        metric_col = metric_col_map[metric_key]
        y_title = metric_label_map[metric_key]

        time_chart = (
            alt.Chart(monthly_state)
            .mark_bar()
            .encode(
                x=alt.X(
                    "month_label:N",
                    title="Month",
                    axis=alt.Axis(labelAngle=-90),
                ),
                y=alt.Y(f"{metric_col}:Q", title=y_title, stack="zero"),
                color=alt.Color("state:N", title="State"),
                tooltip=[
                    "month_label:N",
                    "state:N",
                    "n_incidents:Q",
                    "n_killed:Q",
                    "n_injured:Q",
                ],
            )
            .properties(height=400)
        )

        st.altair_chart(time_chart, width="stretch")

        st.divider()

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
            st.markdown("Incidents by state")
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
            st.altair_chart(incidents_bar, width="stretch")

        with right:
            st.markdown("People killed by state")
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
            st.altair_chart(killed_bar, width="stretch")

        st.divider()

        st.subheader("Participant demographics")

        if p_filtered.empty:
            st.info("No participant level data available for the current filters.")
        else:
            p_age = p_filtered.dropna(subset=["age"]).copy()
            p_age = p_age[(p_age["age"] >= 0) & (p_age["age"] <= 100)]

            col_age, col_gender = st.columns(2)

            with col_age:
                st.markdown("Age distribution for victims and suspects")
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
                    st.altair_chart(age_chart, width="stretch")

            with col_gender:
                st.markdown("Gender by participant role")
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
                    st.altair_chart(gender_chart, width="stretch")

            rel = p_filtered.dropna(subset=["relationship"])
            if not rel.empty:
                st.markdown("Relationship examples such as family or partner")
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
                st.altair_chart(rel_chart, width="stretch")

        st.divider()

        st.subheader("Gun characteristics")

        if guns_filtered.empty:
            st.info("No gun level data available for the current filters.")
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
                st.markdown("Top gun types")
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
                st.altair_chart(gun_type_chart, width="stretch")

            with col_stolen:
                st.markdown("Guns stolen or not stolen")
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
                st.altair_chart(stolen_chart, width="stretch")

            st.markdown("Outcomes by gun type mean and median deaths per incident")

            inc_level = filtered[["incident_id", "n_killed", "n_injured"]].drop_duplicates()
            inc_gun = guns_clean[["incident_id", "gun_type"]].dropna().drop_duplicates()
            merged = inc_gun.merge(inc_level, on="incident_id", how="left")

            gun_stats = (
                merged.groupby("gun_type", as_index=False)
                .agg(
                    incidents=("incident_id", "nunique"),
                    mean_killed=("n_killed", "mean"),
                    median_killed=("n_killed", "median"),
                )
                .sort_values("mean_killed", ascending=False)
            )

            gun_stats["mean_killed"] = gun_stats["mean_killed"].round(2)
            gun_stats["median_killed"] = gun_stats["median_killed"].round(2)

            st.dataframe(
                gun_stats.head(15),
                width="stretch",
                height=350,
            )

        st.divider()

        st.subheader("Map of incidents")

        if "latitude" in filtered.columns and "longitude" in filtered.columns:
            base_cols = [
                "latitude",
                "longitude",
                "date",
                "state",
                "city_or_county",
                "n_killed",
                "n_injured",
                "incident_characteristics",
            ]

            extra_url_cols = ["incident_url", "source_url", "source_url_2", "source_url_3"]
            for c in extra_url_cols:
                if c in filtered.columns:
                    base_cols.append(c)

            map_source = filtered[base_cols].dropna(subset=["latitude", "longitude"])

            if len(map_source) == 0:
                st.info(
                    "There are no incidents with latitude and longitude in the current filters."
                )
            else:
                if len(map_source) > 5000:
                    map_source = map_source.sample(n=5000, random_state=0)

                map_source = map_source.copy()
                map_source["date_str"] = pd.to_datetime(map_source["date"]).dt.strftime("%Y-%m-%d")

                view_state = pdk.ViewState(
                    latitude=float(map_source["latitude"].mean()),
                    longitude=float(map_source["longitude"].mean()),
                    zoom=3,
                    pitch=0,
                )

                map_source["severity"] = map_source["n_killed"] + map_source["n_injured"] + 1

                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_source,
                    get_position='[longitude, latitude]',
                    get_radius="severity * 500",
                    radius_min_pixels=2,
                    radius_max_pixels=15,
                    get_fill_color=[0, 153, 255, 160],
                    pickable=True,
                    auto_highlight=True,
                )

                tooltip_lines = [
                    "{date_str} | {city_or_county}, {state}",
                    "Killed: {n_killed}, Injured: {n_injured}",
                    "Characteristics: {incident_characteristics}",
                ]
                if "incident_url" in map_source.columns:
                    tooltip_lines.append("Incident URL: {incident_url}")
                if "source_url" in map_source.columns:
                    tooltip_lines.append("Source URL: {source_url}")
                if "source_url_2" in map_source.columns:
                    tooltip_lines.append("Source URL 2: {source_url_2}")
                if "source_url_3" in map_source.columns:
                    tooltip_lines.append("Source URL 3: {source_url_3}")

                tooltip = {"text": "\n".join(tooltip_lines)}

                deck = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                )

                st.pydeck_chart(deck, width="stretch")
        else:
            st.info(
                "This data set does not include latitude and longitude so a map is not available."
            )

        st.divider()

        st.subheader("Incident links (click to open)")

        link_cols = [
            c for c in ["incident_url", "source_url", "source_url_2", "source_url_3"]
            if c in filtered.columns
        ]

        if link_cols:
            link_df = (
                filtered[["date", "state", "city_or_county", "n_killed", "n_injured"] + link_cols]
                .sort_values("date", ascending=False)
            )
            st.dataframe(link_df, width="stretch", height=300)
        else:
            st.info("No incident links are available in this dataset.")

        st.divider()

        st.subheader("Raw data for current filters")

        show_cols_base = [
            "date",
            "state",
            "city_or_county",
            "n_killed",
            "n_injured",
            "incident_characteristics",
        ]
        show_cols = [c for c in show_cols_base if c in filtered.columns] + link_cols

        st.dataframe(
            filtered[show_cols].sort_values("date", ascending=False),
            width="stretch",
            height=400,
        )

    except Exception as e:
        st.error(
            "Something went wrong while rendering the dashboard. "
            "Try adjusting the filters or reloading the page.\n\n"
            f"Technical details: {type(e).__name__}: {e}"
        )


if __name__ == "__main__":
    main()