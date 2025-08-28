import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# Constants
STATS_URL = "https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/competitions/8/teams/stats/leaderboard"
STANDINGS_URL = "https://footballapi.pulselive.com/football/standings"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://www.premierleague.com",
    "Referer": "https://www.premierleague.com/fixtures",
    "Accept": "application/json",
    "Accept-Language": "en-GB,en;q=0.9",
}

def get_season_year():
    today = datetime.now()
    if today.month >= 8:
        return str(today.year)
    else:
        return str(today.year - 1)

def fetch_stats(season: str) -> pd.DataFrame:
    params = {"season": season, "_limit": "20"}
    r = requests.get(STATS_URL, params=params, headers=HEADERS, timeout=30)
    data = r.json().get("data", [])
    rows = []
    for row in data:
        tm = row.get("teamMetadata", {})
        s = row.get("stats", {})
        team_id = str(tm.get("id"))
        team_name = tm.get("name")
        stats_row = {"team_id": team_id, "team": team_name}
        stats_row.update(s)
        rows.append(stats_row)
    return pd.DataFrame(rows)

def fetch_standings(season: str) -> pd.DataFrame:
    url = f"https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v5/competitions/8/seasons/{season}/standings"
    params = {"live": "false"}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    entries = r.json().get("tables", [{}])[0].get("entries", [])
    rows = []
    for team in entries:
        team_id = str(team.get("team", {}).get("id"))
        overall = team.get("overall", {})
        played = overall.get("played", None)
        goalsFor = overall.get("goalsFor", None)
        goalsAgainst = overall.get("goalsAgainst", None)
        points = overall.get("points", None)
        rows.append({
            "team_id": team_id,
            "played": played,
            "goalsFor": goalsFor,
            "goalsAgainst": goalsAgainst,
            "points": points
        })
    return pd.DataFrame(rows)

def main():
    season_year = get_season_year()
    # Sidebar for number of matches to show
    match_limit = st.sidebar.number_input("Number of upcoming matches", min_value=1, max_value=50, value=10)

    # Fetch upcoming matches
    MATCHES_URL = "https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches"
    def fetch_matches(season: str, limit: int = 10, stats_df: pd.DataFrame = None) -> pd.DataFrame:
        params = {
            "competition": "8",
            "season": season,
            "period": "PreMatch",
            "_limit": str(limit)
        }
        r = requests.get(MATCHES_URL, params=params, headers=HEADERS, timeout=30)
        data = r.json().get("data", [])
        rows = []
        for m in data:
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            kickoff = m.get("kickoff")
            matchWeek = m.get("matchWeek")
            ground = m.get("ground")
            period = m.get("period")
            home_id = str(home.get("id"))
            away_id = str(away.get("id"))
            row = {
                "Week": matchWeek,
                "Kickoff": kickoff,
                "Home": home.get("name"),
                "HomeId": home_id,
                "Away": away.get("name"),
                "AwayId": away_id,
                "Ground": ground,
                "Period": period
            }
            # Add home/away selected stat only
            if stats_df is not None:
                stat_options = {
                    "xG/90": ("expectedGoals", True),
                        "Poss%": ("possessionPercentage", False),
                    "Pass% Opp Half": ("passingPercentOppHalf", False),
                    "Touch Opp Box/90": ("touchesInOppBox", True)
                }
                stat_keys = list(stat_options.keys())
                # Use selected_stat from main scope
                selected_stat = st.session_state.get("selected_stat", "xG/90")
                metric_key, per90 = stat_options[selected_stat]
                home_stats = stats_df[stats_df["team_id"] == home_id]
                away_stats = stats_df[stats_df["team_id"] == away_id]
                for prefix, stats in [("Home", home_stats), ("Away", away_stats)]:
                    if not stats.empty:
                        value = stats[metric_key].values[0]
                        played = stats["played"].values[0]
                        stat_val = round(float(value) / float(played), 2) if per90 and played else round(float(value), 2)
                        row[f"{prefix} {selected_stat}"] = stat_val
                    else:
                        row[f"{prefix} {selected_stat}"] = None
            rows.append(row)
        df = pd.DataFrame(rows)
        # Convert kickoff to readable format
        if "Kickoff" in df.columns:
            df["Kickoff"] = pd.to_datetime(df["Kickoff"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        if stats_df is not None:
            # Only show selected stat columns
            selected_stat = st.session_state.get("selected_stat", "xG/90")
            display_cols = [
                "Week", "Kickoff", "Home", f"Home {selected_stat}", "Away", f"Away {selected_stat}", "Ground"
            ]
            sorted_df = df[display_cols].sort_values(by="Kickoff")
            numeric_cols = [f"Home {selected_stat}", f"Away {selected_stat}"]
            def highlight_higher(row):
                styles = [''] * len(display_cols)
                home_val = row[f"Home {selected_stat}"]
                away_val = row[f"Away {selected_stat}"]
                home_idx = display_cols.index(f"Home {selected_stat}")
                away_idx = display_cols.index(f"Away {selected_stat}")
                green_style = 'background-color: #b6fcb6; color: black;'
                if pd.notnull(home_val) and pd.notnull(away_val):
                    if home_val > away_val:
                        styles[home_idx] = green_style
                    elif away_val > home_val:
                        styles[away_idx] = green_style
                return styles
            styled_df = sorted_df.style.format({col: '{:.2f}' for col in numeric_cols}).apply(highlight_higher, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
        return df.head(limit)

    # Show schedule table with selected stat only
    st.set_page_config(page_title="EPL Crunch", layout="wide")
    st.markdown("# King of Crunch")
    st.subheader("Upcoming Matches")
    stats_df = fetch_stats(season_year)
    standings_df = fetch_standings(season_year)
    if stats_df.empty or standings_df.empty:
        st.error("Failed to fetch stats or standings.")
        return
    merged = pd.merge(stats_df, standings_df, on="team_id", how="left")
    # Do not rename possessionPercentage to Possession, keep API key for mapping
    # Calculate SH (Shots/90) and SHA (Shots Against/90)
    merged["SH/90"] = merged["totalShots"].astype(float) / merged["played"].astype(float)
    merged["SHA/90"] = merged["totalShotsConceded"].astype(float) / merged["played"].astype(float)
    # Rename goalsFor, goalsAgainst, points columns
    merged = merged.rename(columns={"goalsFor": "GF", "goalsAgainst": "GA", "points": "Pts"})
    # Add Pts% and xGOT%
    merged["Pts%"] = (merged["Pts"].astype(float) / (merged["played"].astype(float) * 3) * 100).round(1)
    merged["SH%"] = (merged["SH/90"].astype(float) / (merged["SH/90"].astype(float) + merged["SHA/90"].astype(float)) * 100).round(1)

    # Stat selection dropdown
    stat_options = {
        "xG/90": ("expectedGoals", True),
        "Poss%": ("possessionPercentage", False),
        "Pass% Opp Half": ("passingPercentOppHalf", False),
        "Touch Opp Box/90": ("touchesInOppBox", True)
    }
    stat_keys = list(stat_options.keys())
    selected_stat = st.selectbox("Select Stat to Display", stat_keys, index=stat_keys.index("xG/90"))
    st.session_state["selected_stat"] = selected_stat


    # Show schedule table with selected stat only
    schedule_df = fetch_matches(season_year, match_limit, stats_df=merged)

    # --- Bottom rankings table (exclude SH%, SH/90, SHA/90; add xG/90) ---
    rank_cols = ["team", "xG/90", "Possession", "GF", "GA", "Pts", "Pts%"]
    # Calculate xG/90 for rankings table
    merged["xG/90"] = merged["expectedGoals"].astype(float) / merged["played"].astype(float)
    st.subheader("Team Rankings")
    sorted_rank = merged.sort_values(by="Pts", ascending=False)
    st.dataframe(sorted_rank[rank_cols], use_container_width=True, height=800)

if __name__ == "__main__":
    main()
