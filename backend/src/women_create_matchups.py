"""
Generate all possible NCAA tournament matchups for 2026 women's basketball
Creates differentials between team and opponent stats
"""
import pandas as pd
import os

print("="*80)
print("CREATING 2026 WOMEN'S MATCHUP DATASET WITH DIFFERENTIALS")
print("="*80 + "\n")

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data', 'women')

print("Loading files...")
matchups_template = pd.read_csv(os.path.join(data_dir, 'bracket_template.csv'))
team_stats = pd.read_csv(os.path.join(data_dir, 'women_teams_enriched.csv'))

print(f"✓ Matchups template: {matchups_template.shape}")
print(f"✓ Team stats: {team_stats.shape}\n")

print("Inverting defensive stats...")
team_stats['adj_de'] = 200 - team_stats['adj_de']
team_stats['efgd_pct'] = 100 - team_stats['efgd_pct']
team_stats['tord'] = 100 - team_stats['tord']
team_stats['drb_pct'] = 100 - team_stats['drb_pct']
team_stats['ftrd'] = 100 - team_stats['ftrd']
team_stats['2pd_pct'] = 100 - team_stats['2pd_pct']
team_stats['3pd_pct'] = 100 - team_stats['3pd_pct']
team_stats['3prd'] = 100 - team_stats['3prd']
print("✓ Defensive stats inverted\n")

print("Adding team names...")
matchups = matchups_template.merge(
    team_stats[['region', 'seed', 'team']],
    left_on=['team_region', 'team_seed'],
    right_on=['region', 'seed'],
    how='left'
).rename(columns={'team': 'team_name'})

matchups = matchups.merge(
    team_stats[['region', 'seed', 'team']],
    left_on=['opp_region', 'opp_seed'],
    right_on=['region', 'seed'],
    how='left',
    suffixes=('', '_opp')
).rename(columns={'team': 'opponent_name'})

matchups = matchups.drop(columns=['region', 'seed', 'region_opp', 'seed_opp'])
print(f"✓ Added team names: {matchups.shape}\n")

print("Joining team stats...")
stat_cols = ['team', 'wab', 'barthag', 'adj_oe', 'adj_de', 'efg_pct', 'efgd_pct', 
             'tor', 'tord', 'orb_pct', 'drb_pct', 'ftr', 'ftrd', '2p_pct', '2pd_pct',
             '3p_pct', '3pd_pct', '3pr', '3prd', 'adj_tempo']

matchups = pd.merge(
    matchups,
    team_stats[stat_cols],
    left_on='team_name',
    right_on='team',
    how='left',
    suffixes=('', '_team')
)
matchups = matchups.drop(columns=['team'])

matchups = pd.merge(
    matchups,
    team_stats[stat_cols],
    left_on='opponent_name',
    right_on='team',
    how='left',
    suffixes=('_team', '_opp')
)
matchups = matchups.drop(columns=['team'])
print(f"✓ Joined team stats: {matchups.shape}\n")

print("Calculating differentials...")
matchups_final = pd.DataFrame()
matchups_final['game_id'] = matchups['game_id']
matchups_final['round'] = matchups['round']
matchups_final['team_region'] = matchups['team_region']
matchups_final['team_seed'] = matchups['team_seed']
matchups_final['team'] = matchups['team_name']
matchups_final['opp_region'] = matchups['opp_region']
matchups_final['opp_seed'] = matchups['opp_seed']
matchups_final['opponent'] = matchups['opponent_name']

matchups_final['wab'] = matchups['wab_team'] - matchups['wab_opp']
matchups_final['barthag'] = matchups['barthag_team'] - matchups['barthag_opp']
matchups_final['adj_oe'] = matchups['adj_oe_team'] - matchups['adj_de_opp']
matchups_final['adj_de'] = matchups['adj_de_team'] - matchups['adj_oe_opp']
matchups_final['efg_pct'] = matchups['efg_pct_team'] - matchups['efgd_pct_opp']
matchups_final['efgd_pct'] = matchups['efgd_pct_team'] - matchups['efg_pct_opp']
matchups_final['tor'] = matchups['tor_team'] - matchups['tord_opp']
matchups_final['tord'] = matchups['tord_team'] - matchups['tor_opp']
matchups_final['orb_pct'] = matchups['orb_pct_team'] - matchups['drb_pct_opp']
matchups_final['drb_pct'] = matchups['drb_pct_team'] - matchups['orb_pct_opp']
matchups_final['ftr'] = matchups['ftr_team'] - matchups['ftrd_opp']
matchups_final['ftrd'] = matchups['ftrd_team'] - matchups['ftr_opp']
matchups_final['2p_pct'] = matchups['2p_pct_team'] - matchups['2pd_pct_opp']
matchups_final['2pd_pct'] = matchups['2pd_pct_team'] - matchups['2p_pct_opp']
matchups_final['3p_pct'] = matchups['3p_pct_team'] - matchups['3pd_pct_opp']
matchups_final['3pd_pct'] = matchups['3pd_pct_team'] - matchups['3p_pct_opp']
matchups_final['3pr'] = matchups['3pr_team'] - matchups['3prd_opp']
matchups_final['3prd'] = matchups['3prd_team'] - matchups['3pr_opp']
matchups_final['adj_tempo'] = matchups['adj_tempo_team'] - matchups['adj_tempo_opp']

print("✓ Calculated differentials\n")

output_path = os.path.join(data_dir, 'women_matchups_current.csv')
matchups_final.to_csv(output_path, index=False)

print(f"✓ Saved to: {output_path}")
print(f"✓ Total matchups: {len(matchups_final)}")
print(f"✓ Total columns: {len(matchups_final.columns)}\n")

print("="*80)
print("COMPLETE!")
print("="*80)