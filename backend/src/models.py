"""
Model loading and inference module for NCAA Tournament predictions
Based on Womens_Composite_Model and Womens_Tiers_Clustering notebooks
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


class NCAAPredictor:
    """
    Handles composite scoring and tier predictions for NCAA Women's Basketball
    """
    
    def __init__(self, historical_data_path='../data'):
        """
        Initialize predictor with historical data for normalization
        
        Args:
            historical_data_path: Path to directory containing historical data
        """
        self.data_path = Path(historical_data_path)
        self.historical_data = None
        self.scalers = {}
        self.weights = {}
        self.percentiles = {}  # Store percentile bounds for scaling
        self.kmeans_model = None
        
        # Feature definitions from notebooks
        self.offensive_vars = ['adj_oe', 'efg_pct', 'tor', 'orb_pct', 'ftr', '2p_pct', '3p_pct', '3pr']
        self.defensive_vars = ['adj_de_inv', 'efgd_pct_inv', 'tord', 'drb_pct', 'ftrd', '2pd_pct_inv', '3pd_pct_inv', '3prd']
        self.overall_vars = ['barthag', 'wab', 'adj_tempo']
        
        # Clustering features
        self.cluster_features = ['adj_oe', 'adj_de', 'barthag', 'efg_pct', 'efgd_pct', 
                                'orb_pct', 'drb_pct', '2p_pct', '2pd_pct', 'wab']
        
    def load_historical_data(self, tournament_file='women_teams_historical.csv', 
                            torvik_file='women_torvik_historical.csv'):
        """
        Load and prepare historical data for model training/normalization
        
        Args:
            tournament_file: CSV with tournament results
            torvik_file: CSV with Torvik advanced statistics
        """
        print("Loading historical data...")
        
        # Load data
        tournament = pd.read_csv(self.data_path / tournament_file)
        torvik = pd.read_csv(self.data_path / torvik_file)
        
        # Merge datasets
        df = tournament.merge(torvik, on='torvik_id', how='left', suffixes=('', '_torvik'))
        
        # Create performance target
        finish_mapping = {
            'First Round': 1,
            'Second Round': 2,
            'Sweet 16': 3,
            'Elite Eight': 4,
            'Final Four': 5,
            'Runner Up': 6,
            'Champion': 7
        }
        df['performance'] = df['finish'].map(finish_mapping)
        
        # Invert defensive features (lower is better → higher is better)
        df['adj_de_inv'] = -df['adj_de']
        df['efgd_pct_inv'] = -df['efgd_pct']
        df['2pd_pct_inv'] = -df['2pd_pct']
        df['3pd_pct_inv'] = -df['3pd_pct']
        
        self.historical_data = df
        print(f"✓ Loaded {len(df)} teams from {df['year'].min()}-{df['year'].max()}")
        
        return df
    
    def train_composite_model(self):
        """
        Train the composite scoring model using historical data
        Calculates feature importance weights for offensive, defensive, and overall scores
        """
        if self.historical_data is None:
            raise ValueError("Must load historical data first using load_historical_data()")
        
        df = self.historical_data
        y = df['performance'].values
        
        print("\n" + "="*80)
        print("TRAINING COMPOSITE MODEL")
        print("="*80)
        
        # Train offensive model
        print("\n1. Building offensive score weights...")
        self.weights['offensive'] = self._calculate_feature_weights(
            df, self.offensive_vars, y
        )
        
        # Calculate percentile bounds for offensive features
        self._calculate_percentile_bounds(df, self.offensive_vars, 'offensive')
        
        # Train defensive model  
        print("2. Building defensive score weights...")
        self.weights['defensive'] = self._calculate_feature_weights(
            df, self.defensive_vars, y
        )
        
        # Calculate percentile bounds for defensive features
        self._calculate_percentile_bounds(df, self.defensive_vars, 'defensive')
        
        # Train overall model
        print("3. Building overall score weights...")
        # First calculate offensive and defensive scores
        df['offensive_score'] = self._calculate_weighted_score(df, self.offensive_vars, self.weights['offensive'], 'offensive')
        df['defensive_score'] = self._calculate_weighted_score(df, self.defensive_vars, self.weights['defensive'], 'defensive')
        
        all_overall_vars = self.overall_vars + ['offensive_score', 'defensive_score']
        overall_df = df[all_overall_vars].copy()
        
        self.weights['overall'] = self._calculate_feature_weights(
            overall_df, all_overall_vars, y
        )
        
        # Calculate percentile bounds for overall features
        self._calculate_percentile_bounds(overall_df, all_overall_vars, 'overall')
        
        print("\n✓ Composite model trained successfully!")
        
    def _calculate_percentile_bounds(self, df, features, score_type):
        """
        Calculate and store 1st and 99th percentile bounds for features
        
        Args:
            df: DataFrame with features
            features: List of feature names
            score_type: 'offensive', 'defensive', or 'overall'
        """
        self.percentiles[score_type] = {}
        for feature in features:
            self.percentiles[score_type][feature] = {
                'p01': df[feature].quantile(0.01),
                'p99': df[feature].quantile(0.99)
            }
    
    def _calculate_feature_weights(self, df, features, target):
        """
        Calculate feature importance weights using correlation + Random Forest
        
        Args:
            df: DataFrame with features
            features: List of feature column names
            target: Target variable array
            
        Returns:
            dict: Feature weights normalized to sum to 1
        """
        # Prepare data
        feature_df = df[features].copy()
        feature_df = feature_df.fillna(feature_df.median())
        X = feature_df.values
        
        # Calculate correlation importance
        correlations = {}
        for feature in features:
            corr = df[feature].corr(pd.Series(target))
            correlations[feature] = abs(corr)
        
        # Calculate Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, target)
        rf_importance = dict(zip(features, rf.feature_importances_))
        
        # Combine 50/50
        combined_weights = {}
        for feature in features:
            corr_norm = correlations[feature] / sum(correlations.values())
            rf_norm = rf_importance[feature] / sum(rf_importance.values())
            combined_weights[feature] = (corr_norm + rf_norm) / 2
        
        # Normalize to sum to 1
        total = sum(combined_weights.values())
        weights = {k: v/total for k, v in combined_weights.items()}
        
        return weights
    
    def _calculate_weighted_score(self, df, features, weights, score_type):
        """
        Calculate weighted score using percentile-based normalization
        
        Args:
            df: DataFrame with features
            features: List of feature names
            weights: Dict of feature weights
            score_type: 'offensive', 'defensive', or 'overall'
            
        Returns:
            Series: Weighted scores
        """
        # Fill missing values
        feature_df = df[features].fillna(df[features].median())
        
        # Normalize each feature to 0-100 scale using percentiles
        normalized = pd.DataFrame(index=df.index)
        
        for feature in features:
            p01 = self.percentiles[score_type][feature]['p01']
            p99 = self.percentiles[score_type][feature]['p99']
            
            # Scale to 0-100, clipping values outside percentile range
            if p99 > p01:  # Avoid division by zero
                normalized[feature] = ((feature_df[feature] - p01) / (p99 - p01) * 100).clip(0, 100)
            else:
                normalized[feature] = 50.0  # If no variance, set to middle
        
        # Calculate weighted sum
        score = sum(normalized[feature] * weights[feature] for feature in features)
        
        return score
    
    def predict_composite_scores(self, team_data):
        """
        Predict offensive, defensive, and overall composite scores for teams
        
        Args:
            team_data: DataFrame with team statistics
            
        Returns:
            DataFrame: Input data with added score columns
        """
        df = team_data.copy()
        
        # Invert defensive features
        df['adj_de_inv'] = -df['adj_de']
        df['efgd_pct_inv'] = -df['efgd_pct']
        df['2pd_pct_inv'] = -df['2pd_pct']
        df['3pd_pct_inv'] = -df['3pd_pct']
        
        # Calculate offensive score
        df['offensive_score'] = self._calculate_weighted_score(
            df, self.offensive_vars, self.weights['offensive'], 'offensive'
        )
        
        # Calculate defensive score
        df['defensive_score'] = self._calculate_weighted_score(
            df, self.defensive_vars, self.weights['defensive'], 'defensive'
        )
        
        # Calculate overall score
        overall_df = df[self.overall_vars + ['offensive_score', 'defensive_score']].copy()
        all_overall_vars = self.overall_vars + ['offensive_score', 'defensive_score']
        df['overall_score'] = self._calculate_weighted_score(
            overall_df, all_overall_vars, self.weights['overall'], 'overall'
        )
        
        # Scale to 1-10 for display
        df['offense'] = (df['offensive_score'] / 10).round(2)
        df['defense'] = (df['defensive_score'] / 10).round(2)
        df['overall'] = (df['overall_score'] / 10).round(2)
        
        return df
    
    def train_tier_model(self):
        """
        Train K-Means clustering model for tier classification
        """
        if self.historical_data is None:
            raise ValueError("Must load historical data first")
        
        print("\n" + "="*80)
        print("TRAINING TIER CLUSTERING MODEL")
        print("="*80 + "\n")
        
        df = self.historical_data
        
        # Prepare clustering features
        cluster_df = df[self.cluster_features].copy()
        
        # Standardize features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)
        
        # Train K-Means with 5 clusters
        self.kmeans_model = KMeans(
            n_clusters=5, 
            init='k-means++', 
            algorithm='lloyd',
            max_iter=500, 
            random_state=123
        )
        self.kmeans_model.fit(scaled_data)
        self.scalers['tier'] = scaler
        
        print("✓ Tier clustering model trained successfully!")
    
    def predict_tiers(self, team_data):
        """
        Predict tier classification for teams
        
        Args:
            team_data: DataFrame with team statistics (must include seed and barthag)
            
        Returns:
            DataFrame: Input data with added 'tier' and 'cluster' columns
        """
        df = team_data.copy()
        
        # Prepare features and scale
        cluster_features_df = df[self.cluster_features].copy()
        scaled_data = self.scalers['tier'].transform(cluster_features_df)
        
        # Predict clusters
        df['cluster'] = self.kmeans_model.predict(scaled_data)
        
        # Calculate barthag rank within year (if year column exists)
        if 'year' in df.columns:
            df['barthag_rtg_rank'] = df.groupby('year')['barthag'].rank(method='min', ascending=False)
        else:
            df['barthag_rtg_rank'] = df['barthag'].rank(method='min', ascending=False)
        
        # Assign tiers based on cluster, rank, and seed
        df['tier'] = self._assign_tier_labels(df)
        
        return df
    
    def _assign_tier_labels(self, df):
        """
        Assign tier labels (S, A, B, C, D) based on cluster, rank, and seed
        Logic from Womens_Tiers_Clustering notebook
        """
        tier = pd.Series('', index=df.index)
        
        # S Tier - Elite teams
        tier[(df['cluster']==2) & (df['barthag_rtg_rank']<=4)] = 'S'
        
        # A Tier - Championship contenders
        tier[(df['cluster']==2) & (df['barthag_rtg_rank']>4) & (df['seed']<=3)] = 'A'
        tier[(df['cluster']==3) & (df['barthag_rtg_rank']<=8) & (df['seed']<=3)] = 'A'
        tier[(df['cluster']==4) & (df['barthag_rtg_rank']<=8) & (df['seed']<=3)] = 'A'
        
        # B Tier - Sweet 16 level
        tier[(df['cluster']==0) & (df['barthag_rtg_rank']<=24) & (df['seed']<=6)] = 'B'
        tier[(df['cluster']==2) & (df['seed']>3) & (df['seed']<=6)] = 'B'
        tier[(df['cluster']==3) & (df['barthag_rtg_rank']<=8) & (df['seed']>3)] = 'B'
        tier[(df['cluster']==4) & (df['barthag_rtg_rank']<=8) & (df['seed']>3)] = 'B'
        
        # C Tier - Round 2 level
        tier[(df['cluster']==0) & (df['barthag_rtg_rank']>24)] = 'C'
        tier[(df['cluster']==1) & (df['seed']<=12)] = 'C'
        tier[(df['cluster']==2) & (df['seed']>6)] = 'C'
        tier[(df['cluster']==3) & (df['barthag_rtg_rank']>8)] = 'C'
        tier[(df['cluster']==4) & (df['barthag_rtg_rank']>8)] = 'C'
        
        # D Tier - First round level
        tier[(df['cluster']==1) & (df['seed']>12)] = 'D'
        
        # Fill any remaining with C
        tier[tier==''] = 'C'
        
        return tier
    
    def batch_predict(self, teams_data):
        """
        Run all predictions for multiple teams
        
        Args:
            teams_data: DataFrame with team statistics
            
        Returns:
            DataFrame: Teams with all predictions added
        """
        # Predict composite scores
        df = self.predict_composite_scores(teams_data)
        
        # Predict tiers
        df = self.predict_tiers(df)
        
        # Add rank based on overall score
        df['rank'] = df['overall_score'].rank(ascending=False, method='min').astype(int)
        
        return df
