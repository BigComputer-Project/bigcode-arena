"""
Elo Rating Calculation Module for BigCodeArena
Contains Bradley-Terry Model with confidence intervals and traditional Elo calculation
"""

import math
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import yaml
import os


def load_model_metadata():
    """Load model metadata from api_config.yaml"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "api_config.yaml")
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        metadata = {}
        for model_key, model_config in config.items():
            if isinstance(model_config, dict):
                model_name = model_config.get("model", model_key)
                metadata[model_name] = {
                    "organization": model_config.get("organization", "Unknown"),
                    "license": model_config.get("license", "Unknown"),
                }
                # Also store with the key name for lookup
                metadata[model_key] = {
                    "organization": model_config.get("organization", "Unknown"),
                    "license": model_config.get("license", "Unknown"),
                }

        return metadata
    except Exception as e:
        print(f"Warning: Could not load model metadata: {e}")
        return {}


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
    """Compute Elo ratings using Bradley-Terry Model with Maximum Likelihood Estimation"""

    # Get all unique models
    all_models = sorted(list(set(df["model_a"].tolist() + df["model_b"].tolist())))
    
    # Create win matrices for each outcome type
    # Initialize empty matrices with float dtype to avoid warnings
    ptbl_a_win = pd.DataFrame(0.0, index=all_models, columns=all_models)
    ptbl_b_win = pd.DataFrame(0.0, index=all_models, columns=all_models)  
    ptbl_tie = pd.DataFrame(0.0, index=all_models, columns=all_models)

    # Count wins for model_a
    model_a_wins = df[df["winner"] == "model_a"]
    if not model_a_wins.empty:
        a_win_counts = model_a_wins.groupby(["model_a", "model_b"]).size()
        for (model_a, model_b), count in a_win_counts.items():
            ptbl_a_win.loc[model_a, model_b] = count

    # Count wins for model_b  
    model_b_wins = df[df["winner"] == "model_b"]
    if not model_b_wins.empty:
        b_win_counts = model_b_wins.groupby(["model_a", "model_b"]).size()
        for (model_a, model_b), count in b_win_counts.items():
            ptbl_b_win.loc[model_a, model_b] = count

    # Count ties
    ties = df[df["winner"].isin(["tie", "tie (bothbad)"])]
    if not ties.empty:
        tie_counts = ties.groupby(["model_a", "model_b"]).size()
        for (model_a, model_b), count in tie_counts.items():
            # For ties, we count 0.5 win for each model
            ptbl_tie.loc[model_a, model_b] = count * 0.5
            ptbl_tie.loc[model_b, model_a] = count * 0.5

    models = pd.Series(np.arange(len(all_models)), index=all_models)
    p = len(models)
    
    # Create training data for logistic regression
    X = []
    Y = []
    sample_weights = []
    
    for model_a in all_models:
        for model_b in all_models:
            if model_a == model_b:
                continue
                
            # Count total games between these models
            a_wins = ptbl_a_win.loc[model_a, model_b]
            b_wins = ptbl_b_win.loc[model_a, model_b] 
            ties = ptbl_tie.loc[model_a, model_b]
            
            total_games = a_wins + b_wins + ties
            if total_games == 0:
                continue
                
            # Create feature vector: difference in model strengths
            x = np.zeros(p)
            x[models[model_a]] = 1.0
            x[models[model_b]] = -1.0
            
            # Add data points for model_a wins
            if a_wins > 0:
                X.append(x)
                Y.append(1)  # model_a wins
                sample_weights.append(a_wins)
            
            # Add data points for model_b wins (model_a loses)
            if b_wins > 0:
                X.append(x)  # same feature vector
                Y.append(0)  # model_a loses
                sample_weights.append(b_wins)
                
            # Add data points for ties - treat as half wins for model_a
            if ties > 0:
                # Add ties as both wins and losses with half weight each
                X.append(x)
                Y.append(1)  # model_a wins (tie counted as win)
                sample_weights.append(ties / 2)
                
                X.append(x)
                Y.append(0)  # model_a loses (tie counted as loss)
                sample_weights.append(ties / 2)

    if len(X) == 0 or len(set(Y)) < 2:
        # Not enough data or no variation in outcomes
        return pd.Series({model: INIT_RATING for model in all_models}).sort_values(ascending=False)

    X = np.array(X)
    Y = np.array(Y)
    sample_weights = np.array(sample_weights)

    # Fit logistic regression
    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6, max_iter=1000)
    lr.fit(X, Y, sample_weight=sample_weights)
    
    # Convert coefficients to Elo ratings
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING


    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round=1000):
    """Get bootstrap results for confidence interval calculation"""

    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        # Bootstrap sample with replacement
        bootstrap_sample = battles.sample(frac=1.0, replace=True)
        try:
            elo_result = func_compute_elo(bootstrap_sample)
            rows.append(elo_result)
        except Exception as e:
            # Skip failed bootstrap samples
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Sort columns by median Elo score (descending)
    return df[df.median().sort_values(ascending=False).index]


def compute_online_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    """Compute Elo ratings for models based on battle results (legacy function for compatibility)"""
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[
        ["model_a", "model_b", "winner"]
    ].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie (bothbad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    # calibrate llama-13b to 800 if it exists
    if "llama-13b" in rating:
        delta = 800 - rating["llama-13b"]
        for model in battles["model_a"].unique():
            rating[model] += delta

    return rating


def calculate_elo_with_confidence_intervals(battles_df, vote_counts):
    """
    Main function to calculate Elo ratings with confidence intervals
    
    Args:
        battles_df (pd.DataFrame): DataFrame with columns ['model_a', 'model_b', 'winner']
        vote_counts (dict): Dictionary with vote counts for each model
        
    Returns:
        tuple: (elo_ratings, confidence_intervals)
    """
    confidence_intervals = {}  # Initialize to avoid uninitialized variable error

    # Check if we have sufficient data for Bradley-Terry model
    if len(battles_df) < 2:
        # Not enough battles, use default ratings
        all_models = set(
            battles_df["model_a"].tolist() + battles_df["model_b"].tolist()
        )
        elo_ratings = pd.Series({model: 1000 for model in all_models})
        confidence_intervals = {model: 0 for model in all_models}
    else:
        try:
            # Use the new Bradley-Terry Model
            elo_ratings = compute_mle_elo(battles_df)

            # Calculate confidence intervals using bootstrap
            if len(battles_df) >= 10:  # Only calculate CI if we have enough data
                try:
                    bootstrap_df = get_bootstrap_result(
                        battles_df, compute_mle_elo, num_round=100
                    )

                    # Calculate 95% confidence intervals
                    if not bootstrap_df.empty:
                        for model in bootstrap_df.columns:
                            scores = bootstrap_df[model].dropna()
                            if len(scores) > 0:
                                lower = scores.quantile(0.025)
                                upper = scores.quantile(0.975)
                                median_score = scores.median()
                                ci_margin = (upper - lower) / 2
                                confidence_intervals[model] = ci_margin
                            else:
                                confidence_intervals[model] = 0
                    else:
                        # Fallback: no confidence intervals
                        for model in elo_ratings.index:
                            confidence_intervals[model] = 0
                except Exception as bootstrap_error:
                    print(
                        f"Bootstrap calculation failed: {bootstrap_error}, skipping confidence intervals"
                    )
                    for model in elo_ratings.index:
                        confidence_intervals[model] = 0
            else:
                # Not enough data for bootstrap, set CI to 0
                for model in elo_ratings.index:
                    confidence_intervals[model] = 0
        except Exception as e:
            # Fallback to old method if Bradley-Terry fails
            print(
                f"Bradley-Terry calculation failed: {e}, falling back to online Elo"
            )
            old_elo_ratings = compute_online_elo(battles_df)
            elo_ratings = pd.Series(old_elo_ratings)
            confidence_intervals = {model: 0 for model in elo_ratings.index}
    return elo_ratings, confidence_intervals


def create_ranking_dataframe(elo_ratings, confidence_intervals, vote_counts):
    """
    Create ranking DataFrame with all necessary columns

    Args:
        elo_ratings (pd.Series): Elo ratings for each model
        confidence_intervals (dict): Confidence interval margins for each model
        vote_counts (dict): Vote counts for each model

    Returns:
        pd.DataFrame: Ranking table with columns [Rank, Model, Score, 95% CI (±), Votes, Organization, License]
    """
    # Load model metadata
    metadata = load_model_metadata()

    # Create ranking list with Elo ratings and confidence intervals
    ranking_list = []
    for model in elo_ratings.index:
        ci_margin = confidence_intervals.get(model, 0)

        # Get metadata for this model
        model_metadata = metadata.get(model, {})
        organization = model_metadata.get("organization", "Unknown")
        license_type = model_metadata.get("license", "Unknown")

        ranking_list.append(
            {
                "Model": model,
                "Score": round(elo_ratings[model], 1),
                "95% CI (±)": round(ci_margin, 1) if ci_margin > 0 else "-",
                "Votes": vote_counts[model],
                "Organization": organization,
                "License": license_type,
            }
        )

    # Sort by Elo rating (highest first)
    ranking_df = pd.DataFrame(ranking_list).sort_values("Score", ascending=False)
    ranking_df["Rank"] = range(1, len(ranking_df) + 1)

    # Reorder columns
    ranking_df = ranking_df[
        ["Rank", "Model", "Score", "95% CI (±)", "Votes", "Organization", "License"]
    ]

    return ranking_df
