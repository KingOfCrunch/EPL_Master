import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import asyncio
import aiohttp
from understat import Understat
import difflib
import numpy as np
import plotly.graph_objects as go
import json

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

def fuzzy_match_team(epl_team, understat_teams):
    # Try to match EPL team name to Understat team name
    match = difflib.get_close_matches(epl_team, understat_teams, n=1, cutoff=0.7)
    return match[0] if match else None

def get_understat_season():
    # Understat season format: '2024' for 2024/25
    today = datetime.now()
    if today.month >= 8:
        return str(today.year)
    else:
        return str(today.year - 1)

async def fetch_understat_matches(season):
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        matches = await understat.get_league_results("EPL", season)
        return matches

def find_understat_match(epl_row, understat_matches):
    # Match if dates are within 2 days and fuzzy match team names
    epl_date = pd.to_datetime(epl_row['Kickoff'], errors='coerce')
    epl_home = epl_row['Home']
    epl_away = epl_row['Away']
    best_match = None
    for m in understat_matches:
        try:
            match_date = pd.to_datetime(m['datetime'], errors='coerce')
            home_team = m['h']['title']
            away_team = m['a']['title']
            date_diff = abs((match_date - epl_date).days)
            home_match = fuzzy_match_team(epl_home, [home_team])
            away_match = fuzzy_match_team(epl_away, [away_team])
            if date_diff <= 2 and home_match and away_match:
                # Prefer exact date match, but accept within 2 days
                if date_diff == 0:
                    return m
                best_match = m
        except Exception as ex:
            continue
    if not best_match:
        # Debug output for unmatched rows
        print(f"No Understat match for EPL: {epl_home} vs {epl_away} on {epl_date}")
    return best_match

def main():
    season_year = get_season_year()
    # Sidebar for match period selection
    match_period = st.sidebar.selectbox(
        "View", ["Upcoming", "Completed"], index=0,
        help="Select 'Upcoming' for pre-match fixtures, 'Completed' for finished matches."
    )
    # Sidebar for number of matches to show
    if match_period == "Upcoming":
        match_limit = st.sidebar.number_input("Number of upcoming matches", min_value=1, max_value=50, value=10)
        period_val = "PreMatch"
    else:
        match_limit = None  # Show all completed matches
        period_val = "FullTime"

    MATCHES_URL = "https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches"
    def fetch_upcoming_matches(season: str, limit: int = 10, stats_df: pd.DataFrame = None) -> pd.DataFrame:
        # ...existing code for upcoming matches...
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
            period_api = m.get("period")
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
                "Period": period_api
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

    def fetch_completed_matches(season: str, limit: int = 100) -> pd.DataFrame:
        params = {
            "competition": "8",
            "season": season,
            "period": "FullTime"
        }
        r = requests.get(MATCHES_URL, params=params, headers=HEADERS, timeout=30)
        data = r.json().get("data", [])
        rows = []
        for m in data:
            home = m.get("homeTeam", {})
            away = m.get("awayTeam", {})
            kickoff = m.get("kickoff")
            matchWeek = m.get("matchWeek")
            home_score = home.get("score")
            away_score = away.get("score")
            home_logo = home.get("crestUrl") if "crestUrl" in home else None
            away_logo = away.get("crestUrl") if "crestUrl" in away else None
            match_id = m.get("matchId")
            row = {
                "Week": matchWeek,
                "Kickoff": kickoff,
                "Home": home.get("name"),
                "HomeScore": home_score,
                "HomeLogo": home_logo,
                "Away": away.get("name"),
                "AwayScore": away_score,
                "AwayLogo": away_logo,
                "matchId": match_id
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        # Convert kickoff to readable format
        if "Kickoff" in df.columns:
            df["Kickoff"] = pd.to_datetime(df["Kickoff"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
        return df

    st.set_page_config(page_title="EPL Crunch", layout="wide")
    st.markdown("# ⚽ Premier League Fixture Analysis")
    if match_period == "Upcoming":
        st.subheader("Upcoming Matches")
        stats_df = fetch_stats(season_year)
        standings_df = fetch_standings(season_year)
        if stats_df.empty or standings_df.empty:
            st.error("Failed to fetch stats or standings.")
            return
        merged = pd.merge(stats_df, standings_df, on="team_id", how="left")
        merged = merged.rename(columns={"goalsFor": "GF", "goalsAgainst": "GA", "points": "Pts"})
        merged["Pts%"] = (merged["Pts"].astype(float) / (merged["played"].astype(float) * 3) * 100).round(1)

        stat_options = {
            "xG/90": ("expectedGoals", True),
            "Poss%": ("possessionPercentage", False),
            "Pass% Opp Half": ("passingPercentOppHalf", False),
            "Touch Opp Box/90": ("touchesInOppBox", True)
        }
        stat_keys = list(stat_options.keys())
        selected_stat = st.selectbox("Select Stat to Display", stat_keys, index=stat_keys.index("xG/90"))
        st.session_state["selected_stat"] = selected_stat

        schedule_df = fetch_upcoming_matches(season_year, match_limit, stats_df=merged)
        merged["xG/90"] = (merged["expectedGoals"].astype(float) / merged["played"].astype(float)).round(2)
        if "possessionPercentage" in merged.columns:
            merged = merged.rename(columns={"possessionPercentage": "Possession"})
        rank_cols = ["team", "xG/90", "Possession", "GF", "GA", "Pts", "Pts%"]
        st.subheader("Team Rankings")
        sorted_rank = merged.sort_values(by="Pts", ascending=False)
        st.dataframe(sorted_rank[rank_cols], use_container_width=True, height=800)
    else:
        st.subheader("Completed Matches")
        understat_season = get_understat_season()
        # Fetch Understat matches for the season
        try:
            understat_matches = asyncio.run(fetch_understat_matches(understat_season))
        except Exception as e:
            st.error(f"Failed to fetch Understat matches: {e}")
            understat_matches = []
        if not understat_matches:
            st.info("No completed matches found.")
            return
        # Prepare DataFrame from Understat matches
        rows = []
        for m in understat_matches:
            home = m['h']['title']
            away = m['a']['title']
            kickoff = m.get('datetime')
            match_id = m.get('id')
            week = m.get('week', None)
            home_score = m.get('goals', {}).get('h') if m.get('goals') else m.get('home_goals', None)
            away_score = m.get('goals', {}).get('a') if m.get('goals') else m.get('away_goals', None)
            row = {
                "Week": week,
                "Kickoff": kickoff,
                "Home": home,
                "HomeScore": home_score,
                "Away": away,
                "AwayScore": away_score,
                "UnderstatId": match_id,
                "UnderstatObj": m
            }
            rows.append(row)
        matches_df = pd.DataFrame(rows)
        if "Kickoff" in matches_df.columns:
            matches_df["Kickoff"] = pd.to_datetime(matches_df["Kickoff"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    # Check for query param to show stats page
    query_params = st.query_params
    stats_match_id = query_params.get("stats_match_id", None)
    stats_match_row = None
    if stats_match_id:
        # Find the match row by UnderstatId directly
        for idx, row in matches_df.iterrows():
            if str(row["UnderstatId"]) == str(stats_match_id):
                stats_match_row = row
                break
        if stats_match_row is None:
            st.info(f"No match found for stats_match_id={stats_match_id}")
    if stats_match_row is not None and stats_match_row['UnderstatObj']:
        # Only show stats view, hide match list
        st.markdown("# 📊 Match Stats", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)
        date_str = stats_match_row["Kickoff"] if stats_match_row["Kickoff"] else ""
        home = stats_match_row["Home"]
        away = stats_match_row["Away"]
        home_score = stats_match_row["HomeScore"] if stats_match_row["HomeScore"] is not None else "?"
        away_score = stats_match_row["AwayScore"] if stats_match_row["AwayScore"] is not None else "?"
        home_logo_html = ""
        away_logo_html = ""
        st.markdown(f"<div style='margin-bottom:8px;'><span style='color:#888;'>{date_str}</span></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:center;margin-bottom:8px;'>"
            f"<div style='flex:1;text-align:right;font-weight:bold;font-size:1.2em;'>"
            f"<b>{home_logo_html}{home}</b>"
            f"</div>"
            f"<div style='flex:0.5;text-align:center;font-size:1.3em;padding:0 16px;'>"
            f"<span style='background:#f3f3f3;border-radius:8px;padding:4px 12px;'><b>{home_score} - {away_score}</b></span>"
            f"</div>"
            f"<div style='flex:1;text-align:left;font-weight:bold;font-size:1.2em;'>"
            f"<b>{away}{away_logo_html}</b>"
            f"</div>"
            f"</div>", unsafe_allow_html=True)
        xg_home = float(stats_match_row['UnderstatObj'].get('xG', {}).get('h', 0))
        xg_away = float(stats_match_row['UnderstatObj'].get('xG', {}).get('a', 0))
        st.markdown(f"<div style='text-align: center; font-size: 1em; margin-top: 4px; margin-bottom: 18px;'>xG: {xg_home:.2f} – {xg_away:.2f}</div>", unsafe_allow_html=True)
        # Added extra margin below xG score for spacing
        def simulate_match(xg_home, xg_away, n_sim=1000):
            home_goals = np.random.poisson(xg_home, n_sim)
            away_goals = np.random.poisson(xg_away, n_sim)
            home_win = np.sum(home_goals > away_goals)
            draw = np.sum(home_goals == away_goals)
            away_win = np.sum(home_goals < away_goals)
            return home_win, draw, away_win
        home_win, draw, away_win = simulate_match(xg_home, xg_away, 1000)
        total = home_win + draw + away_win
        outcome_labels = [f"{home} Win", "Draw", f"{away} Win"]
        outcome_values = [home_win, draw, away_win]
        outcome_percent = [v / total * 100 for v in outcome_values]
        # Removed custom Outcome Key label
        st.markdown("<div style='text-align:center; font-size:1.2em; font-weight:bold; margin-bottom:4px;'>Simulated Outcomes (1000 runs)</div>", unsafe_allow_html=True)
        donut_fig = go.Figure(data=[go.Pie(labels=outcome_labels, values=outcome_percent, hole=0.5, marker=dict(colors=["#b6fcb6", "#f3f3f3", "#fcb6b6"]))])
        donut_fig.update_layout(
            title_text="",
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1)
        )
        st.plotly_chart(donut_fig, use_container_width=True)
        async def fetch_shots(match_id):
            async with aiohttp.ClientSession() as session:
                understat = Understat(session)
                shots = await understat.get_match_shots(int(match_id))
                return shots
        def get_shots_sync(match_id):
            try:
                return asyncio.run(fetch_shots(match_id))
            except RuntimeError:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(fetch_shots(match_id))
        try:
            with st.spinner("Loading Understat shots..."):
                shots_data = get_shots_sync(stats_match_id)
                all_shots = []
                for side in ['h', 'a']:
                    for shot in shots_data.get(side, []):
                        all_shots.append({
                            'minute': int(shot['minute']),
                            'player': shot['player'],
                            'team': shot['h_team'] if side == 'h' else shot['a_team'],
                            'xG': float(shot['xG']),
                            'result': shot['result'],
                            'shotType': shot['shotType'],
                            'situation': shot['situation'],
                            'side': side,
                            'time': shot['minute'],
                        })
                all_shots = sorted(all_shots, key=lambda x: x['minute'])
                has_penalty = any(shot['situation'] == 'Penalty' for shot in all_shots)
                if has_penalty:
                    penalty_home_xg = sum(shot['xG'] for shot in all_shots if shot['situation'] == 'Penalty' and shot['side'] == 'h')
                    penalty_away_xg = sum(shot['xG'] for shot in all_shots if shot['situation'] == 'Penalty' and shot['side'] == 'a')
                    xg_home_nopen = max(xg_home - penalty_home_xg, 0)
                    xg_away_nopen = max(xg_away - penalty_away_xg, 0)
                    home_win2, draw2, away_win2 = simulate_match(xg_home_nopen, xg_away_nopen, 1000)
                    total2 = home_win2 + draw2 + away_win2
                    donut2_labels = outcome_labels
                    donut2_values = [home_win2, draw2, away_win2]
                    donut2_percent = [v / total2 * 100 for v in donut2_values]
                    # Removed custom Outcome Key label
                    st.markdown("<div style='text-align:center; font-size:1.2em; font-weight:bold; margin-bottom:4px;'>Simulated Outcomes (No Penalties)</div>", unsafe_allow_html=True)
                    donut2_fig = go.Figure(data=[go.Pie(labels=donut2_labels, values=donut2_percent, hole=0.5, marker=dict(colors=["#b6fcb6", "#f3f3f3", "#fcb6b6"]))])
                    donut2_fig.update_layout(
                        title_text="",
                        showlegend=True,
                        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1)
                    )
                    st.plotly_chart(donut2_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to fetch Understat shots: {e}")
        st.button("⬅️ Back to Results", on_click=lambda: st.query_params.clear())
    else:
        # Only show match list if not viewing stats
        all_teams = sorted(set(matches_df['Home']).union(set(matches_df['Away'])))
        team_options = ['All Teams'] + all_teams
        selected_team = st.selectbox('Filter matches by team (home or away):', team_options, index=0)
        if selected_team == 'All Teams':
            filtered_matches = matches_df
        else:
            filtered_matches = matches_df[(matches_df['Home'] == selected_team) | (matches_df['Away'] == selected_team)]
        sorted_matches = filtered_matches.sort_values(by="Kickoff", ascending=False)
        st.markdown("""
        <style>
        div[data-testid='stButton'] button {
            background-color: #0074cc !important;
            color: #fff !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 1em !important;
            border-radius: 6px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        </style>
        """, unsafe_allow_html=True)
        for idx, row in sorted_matches.iterrows():
            date_str = row["Kickoff"] if row["Kickoff"] else ""
            home = row["Home"]
            away = row["Away"]
            home_score = row["HomeScore"] if row["HomeScore"] is not None else "?"
            away_score = row["AwayScore"] if row["AwayScore"] is not None else "?"
            understat_id = row["UnderstatId"]
            st.markdown(f"<div style='margin-bottom:8px; text-align:center;'><span style='color:#888;'>{date_str}</span></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='display:flex; align-items:center; justify-content:center; margin-bottom:8px;'>
                <div style='flex:1; text-align:right; font-weight:bold; font-size:1.2em;'>{home}</div>
                <div style='flex:0.3; text-align:center; font-size:1.1em; background:#f3f3f3; border-radius:8px; padding:4px 12px; margin:0 4px; min-width:70px; max-width:90px;'><b>{home_score} - {away_score}</b></div>
                <div style='flex:1; text-align:left; font-weight:bold; font-size:1.2em;'>{away}</div>
            </div>
            """, unsafe_allow_html=True)
            if understat_id:
                col1, col2, col3 = st.columns([1,1,1])
                with col2:
                    if st.button("View Stats", key=f"view_stats_{understat_id}", use_container_width=True):
                        st.query_params["stats_match_id"] = str(understat_id)
                        st.rerun()
            else:
                st.markdown("<div style='text-align:center;'><span style='color:#888;'>No Understat stats available</span></div>", unsafe_allow_html=True)
            st.markdown("<hr style='border:1px solid #eee; margin: 24px 0;'>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()