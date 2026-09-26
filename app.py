import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="NHL Fantasy Draft Tool", layout="wide")

# Initialize session state for draft tracking
if 'drafted' not in st.session_state:
    st.session_state['drafted'] = {} # dict mapping playerName -> 'My Team' or 'Opponent'

@st.cache_data
def load_data():
    skaters = pd.read_csv('skaters.csv')
    goalies = pd.read_csv('goalies.csv')
    try:
        adp = pd.read_csv('adp.csv')
    except:
        adp = pd.DataFrame(columns=['Name', 'ADP'])
    return skaters, goalies, adp

def calculate_skater_fpts(df):
    df = df.fillna(0)
    ppa = df['ppPoints'] - df['ppGoals']
    sha = df['shPoints'] - df['shGoals']
    fpts = (df['goals'] * 2 + df['assists'] * 1 + df['ppGoals'] * 0.5 + ppa * 0.5 + 
            df['shGoals'] * 1 + sha * 1 + df['shots'] * 0.2 + df['hits'] * 0.2 + df['blockedShots'] * 0.2)
    return fpts

def calculate_goalie_fpts(df):
    df = df.fillna(0)
    fpts = (df['wins'] * 3 + df['goalsAgainst'] * -0.5 + df['saves'] * 0.1 + df['shutouts'] * 4)
    return fpts

def get_projections(skaters, goalies, w1, w2, w3, sh_mult, league_size, adp):
    skaters = skaters.copy()
    goalies = goalies.copy()
    
    skaters['fpts'] = calculate_skater_fpts(skaters)
    goalies['fpts'] = calculate_goalie_fpts(goalies)
    
    def process_player(df, is_goalie=False):
        total_gp = df[~df['season'].str.endswith('_2H')]['gamesPlayed'].sum()
        if total_gp < 10:
            return 0
        y1 = df[df['season'] == '20232024']
        y2 = df[df['season'] == '20242025']
        y3 = df[df['season'] == '20252026']
        y3_sh = df[df['season'] == '20252026_2H']
        
        fppg_y1 = y1['fpts'].sum() / y1['gamesPlayed'].sum() if not y1.empty and y1['gamesPlayed'].sum() > 0 else np.nan
        fppg_y2 = y2['fpts'].sum() / y2['gamesPlayed'].sum() if not y2.empty and y2['gamesPlayed'].sum() > 0 else np.nan
        
        if not y3.empty and y3['gamesPlayed'].sum() > 0:
            fs_games = y3['gamesPlayed'].sum()
            fs_fpts = y3['fpts'].sum()
            if not y3_sh.empty and y3_sh['gamesPlayed'].sum() > 0:
                sh_games = y3_sh['gamesPlayed'].sum()
                sh_fpts = y3_sh['fpts'].sum()
                fh_games = max(0, fs_games - sh_games)
                fh_fpts = fs_fpts - sh_fpts
                
                fppg_fh = fh_fpts / fh_games if fh_games > 0 else 0
                fppg_sh = sh_fpts / sh_games if sh_games > 0 else 0
                
                fppg_y3 = (fppg_fh * fh_games + fppg_sh * sh_mult * sh_games) / fs_games
            else:
                fppg_y3 = fs_fpts / fs_games
                fppg_sh = np.nan
        else:
            fppg_y3 = np.nan
            fppg_sh = np.nan
            
        weights = []
        vals = []
        if not np.isnan(fppg_y1):
            weights.append(w1)
            vals.append(fppg_y1)
        if not np.isnan(fppg_y2):
            weights.append(w2)
            vals.append(fppg_y2)
        if not np.isnan(fppg_y3):
            weights.append(w3)
            vals.append(fppg_y3)
            
        if not weights:
            return 0
            
        proj_fppg = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        proj_games = 55 if is_goalie else 82
        return proj_fppg * proj_games, proj_fppg, fppg_y1, fppg_y2, fppg_y3, fppg_sh

    all_projs = []
    for pid, group in skaters.groupby(['playerId', 'skaterFullName', 'positionCode']):
        res = process_player(group, is_goalie=False)
        if res and res[0] > 0:
            all_projs.append({
                'Name': group['skaterFullName'].iloc[0],
                'Position': group['positionCode'].iloc[0],
                'Projected Total Points': round(res[0], 2),
                'Projected Points/Game': round(res[1], 2),
                '23-24 FP/G': round(res[2], 2) if pd.notnull(res[2]) else None,
                '24-25 FP/G': round(res[3], 2) if pd.notnull(res[3]) else None,
                '25-26 FP/G': round(res[4], 2) if pd.notnull(res[4]) else None,
                '25-26 2H FP/G': round(res[5], 2) if pd.notnull(res[5]) else None
            })
            
    for pid, group in goalies.groupby(['playerId', 'goalieFullName']):
        res = process_player(group, is_goalie=True)
        if res and res[0] > 0:
            all_projs.append({
                'Name': group['goalieFullName'].iloc[0],
                'Position': 'G',
                'Projected Total Points': round(res[0], 2),
                'Projected Points/Game': round(res[1], 2),
                '23-24 FP/G': round(res[2], 2) if pd.notnull(res[2]) else None,
                '24-25 FP/G': round(res[3], 2) if pd.notnull(res[3]) else None,
                '25-26 FP/G': round(res[4], 2) if pd.notnull(res[4]) else None,
                '25-26 2H FP/G': round(res[5], 2) if pd.notnull(res[5]) else None
            })
            
    all_players_df = pd.DataFrame(all_projs).sort_values('Projected Total Points', ascending=False).reset_index(drop=True)
    
    # Calculate VORP
    replacement_ranks = {
        'C': league_size * 2,
        'L': league_size * 2,
        'R': league_size * 2,
        'D': league_size * 4,
        'G': league_size * 2
    }
    
    replacement_levels = {}
    for pos, rank in replacement_ranks.items():
        pos_df = all_players_df[all_players_df['Position'] == pos]
        if len(pos_df) > rank:
            replacement_levels[pos] = pos_df.iloc[rank-1]['Projected Total Points']
        else:
            replacement_levels[pos] = 0 if pos_df.empty else pos_df.iloc[-1]['Projected Total Points']
            
    all_players_df['VORP'] = all_players_df.apply(lambda row: round(row['Projected Total Points'] - replacement_levels.get(row['Position'], 0), 2), axis=1)
    all_players_df = all_players_df.sort_values('VORP', ascending=False).reset_index(drop=True)
    all_players_df['Rank'] = all_players_df.index + 1
    
    # Merge ADP
    all_players_df = pd.merge(all_players_df, adp, on='Name', how='left')
    all_players_df['Value (ADP - Rank)'] = all_players_df.apply(
        lambda row: round(row['ADP'] - row['Rank'], 1) if pd.notnull(row['ADP']) else None, axis=1
    )
    
    return all_players_df

st.title("NHL Fantasy Draft Tool")

# Sidebar for toggles
st.sidebar.header("League Settings")
league_size = st.sidebar.number_input("League Size (Number of Teams)", min_value=4, max_value=32, value=10)

st.sidebar.header("Weight Settings (Must sum to 100%)")
w3_pct = st.sidebar.slider("Weight: 2025-2026 Season (%)", 0, 100, 80)
rem_w2 = 100 - w3_pct
w2_pct = st.sidebar.slider("Weight: 2024-2025 Season (%)", 0, rem_w2, min(15, rem_w2))
w1_pct = 100 - w3_pct - w2_pct
st.sidebar.markdown(f"**Weight: 2023-2024 Season:** {w1_pct}%")

w1 = w1_pct / 100.0
w2 = w2_pct / 100.0
w3 = w3_pct / 100.0

sh_mult = st.sidebar.slider("2nd Half Bonus Multiplier", 1.0, 2.0, 1.0)

# Load data
skaters_raw, goalies_raw, adp_raw = load_data()

# Calculate projections
all_players_df = get_projections(skaters_raw, goalies_raw, w1, w2, w3, sh_mult, league_size, adp_raw)

# Draft Tabs
tab_available, tab_my_team, tab_draft_board = st.tabs(["Available Players", "My Team", "Full Draft Board"])

with tab_available:
    st.header("Available Players (Ranked by VORP)")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        search_query = st.text_input("Search for a player by name:")
    with col2:
        pos_filter = st.multiselect("Filter by Position:", ["C", "L", "R", "D", "G"])

    # Filter out drafted players
    available_df = all_players_df[~all_players_df['Name'].isin(st.session_state['drafted'].keys())]
    
    if search_query:
        available_df = available_df[available_df['Name'].str.contains(search_query, case=False, na=False)]
    if pos_filter:
        available_df = available_df[available_df['Position'].isin(pos_filter)]

    st.markdown("Select a row to Draft them or view stats!")

    event = st.dataframe(
        available_df[['Rank', 'Name', 'Position', 'VORP', 'ADP', 'Value (ADP - Rank)', 'Projected Points/Game', '23-24 FP/G', '24-25 FP/G', '25-26 FP/G', '25-26 2H FP/G']], 
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row"
    )

    if event and getattr(event, "selection", None) and event.selection.get("rows"):
        selected_idx = event.selection["rows"][0]
        selected_name = available_df.iloc[selected_idx]['Name']
        
        st.markdown(f"### Player Action: {selected_name}")
        
        colA, colB = st.columns([1, 1])
        with colA:
            if st.button(f"Draft {selected_name} to My Team"):
                st.session_state['drafted'][selected_name] = 'My Team'
                st.rerun()
        with colB:
            if st.button(f"{selected_name} Drafted by Opponent"):
                st.session_state['drafted'][selected_name] = 'Opponent'
                st.rerun()
        
        # Get player stats from raw data
        s_stats = skaters_raw[skaters_raw['skaterFullName'] == selected_name]
        g_stats = goalies_raw[goalies_raw['goalieFullName'] == selected_name]
        
        if not s_stats.empty:
            s_stats = s_stats.copy()
            s_stats['fpts'] = calculate_skater_fpts(s_stats)
            s_stats['FPPG'] = round(s_stats['fpts'] / s_stats['gamesPlayed'].clip(lower=1), 2)
            st.dataframe(s_stats.drop(columns=['playerId', 'skaterFullName', 'positionCode', 'fpts']), use_container_width=True)
        elif not g_stats.empty:
            g_stats = g_stats.copy()
            g_stats['fpts'] = calculate_goalie_fpts(g_stats)
            g_stats['FPPG'] = round(g_stats['fpts'] / g_stats['gamesPlayed'].clip(lower=1), 2)
            st.dataframe(g_stats.drop(columns=['playerId', 'goalieFullName', 'fpts']), use_container_width=True)

with tab_my_team:
    st.header("My Roster")
    my_team_names = [name for name, status in st.session_state['drafted'].items() if status == 'My Team']
    my_team_df = all_players_df[all_players_df['Name'].isin(my_team_names)]
    
    st.dataframe(my_team_df[['Name', 'Position', 'VORP', 'Projected Total Points']], use_container_width=True)
    
    if len(my_team_df) > 0:
        st.markdown(f"**Total Projected Points:** {my_team_df['Projected Total Points'].sum():.2f}")
        
    st.markdown("### Undo Draft Action")
    undo_name = st.selectbox("Select player to remove from draft", ["None"] + list(st.session_state['drafted'].keys()))
    if undo_name != "None" and st.button("Undo Draft"):
        del st.session_state['drafted'][undo_name]
        st.rerun()

with tab_draft_board:
    st.header("Draft Tracker")
    drafted_names = list(st.session_state['drafted'].keys())
    drafted_df = all_players_df[all_players_df['Name'].isin(drafted_names)].copy()
    
    if not drafted_df.empty:
        drafted_df['Drafted By'] = drafted_df['Name'].map(st.session_state['drafted'])
        st.dataframe(drafted_df[['Name', 'Position', 'Drafted By', 'Projected Total Points']], use_container_width=True)
    else:
        st.markdown("No players drafted yet.")
