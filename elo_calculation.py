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


def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None):
    """Compute Elo ratings using Bradley-Terry Model with Maximum Likelihood Estimation"""

    # Create pivot tables for different outcomes
    # Check if we have any model_a wins
    model_a_wins = df[df["winner"] == "model_a"]
    if model_a_wins.empty:
        # Get all unique models to create empty pivot table
        all_models = list(set(df["model_a"].tolist() + df["model_b"].tolist()))
        ptbl_a_win = pd.DataFrame(0, index=all_models, columns=all_models)
    else:
        ptbl_a_win = pd.pivot_table(
            model_a_wins,
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )

    # Handle ties - if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    # Check if we have any model_b wins
    model_b_wins = df[df["winner"] == "model_b"]
    if model_b_wins.empty:
        # Use same structure as ptbl_a_win
        ptbl_b_win = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_b_win = pd.pivot_table(
            model_b_wins,
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        # Ensure same index/columns as ptbl_a_win
        ptbl_b_win = ptbl_b_win.reindex(
            index=ptbl_a_win.index, columns=ptbl_a_win.columns, fill_value=0
        )

    # Combine all outcomes: A wins count as 2, B wins count as 2, ties count as 1 each
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue

            # Model A vs Model B
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            # Model B vs Model A
            X[cur_row + 1, models[m_a]] = -math.log(BASE)
            X[cur_row + 1, models[m_b]] = +math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2

    X = X[:cur_row]
    Y = Y[:cur_row]

    # Check if we have enough data for fitting
    if cur_row == 0 or len(set(Y)) < 2:
        # Not enough data or no variation in outcomes, return default ratings
        return pd.Series({model: INIT_RATING for model in models.index}).sort_values(
            ascending=False
        )

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # Calibrate to mixtral-8x7b-instruct-v0.1 if it exists
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]

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
        pd.DataFrame: Ranking table with columns [Rank, Model, Score, 95% CI (±), Votes]
    """
    # Create ranking list with Elo ratings and confidence intervals
    ranking_list = []
    for model in elo_ratings.index:
        ci_margin = confidence_intervals.get(model, 0)
        ranking_list.append(
            {
                "Model": model,
                "Score": round(elo_ratings[model], 1),
                "95% CI (±)": round(ci_margin, 1) if ci_margin > 0 else "-",
                "Votes": vote_counts[model],
            }
        )

    # Sort by Elo rating (highest first)
    ranking_df = pd.DataFrame(ranking_list).sort_values("Score", ascending=False)
    ranking_df["Rank"] = range(1, len(ranking_df) + 1)

    # Reorder columns
    ranking_df = ranking_df[["Rank", "Model", "Score", "95% CI (±)", "Votes"]]

    return ranking_df
