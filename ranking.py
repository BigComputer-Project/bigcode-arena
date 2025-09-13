"""
Ranking module for BigCodeArena
Handles model leaderboard functionality and data management
"""

import gradio as gr
import pandas as pd
import datetime
import os
from collections import defaultdict
from datasets import Dataset, load_dataset

# Import Elo calculation utilities
from elo_calculation import (
    calculate_elo_with_confidence_intervals,
    create_ranking_dataframe,
)

# HuggingFace dataset configuration
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
REFRESH_TIME = os.getenv("REFRESH_TIME") or 60*60*12 # 12 hours by default

# Global ranking data cache
ranking_data = None
ranking_last_updated = None


def load_ranking_data(hf_token=None, force_reload=False):
    """Load and calculate ranking data from HuggingFace dataset"""
    global ranking_data, ranking_last_updated

    try:
        # Use global token if not provided
        token = hf_token or HF_TOKEN
        
        if not token:
            return pd.DataFrame()

        if not HF_DATASET_NAME:
            return pd.DataFrame()

        # Load dataset - force download if requested
        if force_reload:
            # Force download from remote, ignore cache
            dataset = load_dataset(
                HF_DATASET_NAME,
                split="train",
                token=token,
                download_mode="force_redownload",
            )
        else:
            dataset = load_dataset(HF_DATASET_NAME, split="train", token=token)
        # Convert to pandas DataFrame - handle both Dataset and DatasetDict
        if hasattr(dataset, "to_pandas"):
            df = dataset.to_pandas()
        else:
            df = pd.DataFrame(dataset)

        if df.empty:
            return pd.DataFrame()

        # Convert vote format for Elo calculation and count votes
        battle_data = []
        vote_counts = defaultdict(int)

        for _, row in df.iterrows():
            model_a = row["model_a"]
            model_b = row["model_b"]
            vote = row["vote"]

            # Convert vote to winner format for Elo
            if vote == "left":  # Model A wins
                winner = "model_a"
            elif vote == "right":  # Model B wins
                winner = "model_b"
            elif vote == "tie":
                winner = "tie"
            elif vote == "both_bad":
                winner = "tie (bothbad)"
            else:
                continue  # Skip invalid votes

            battle_data.append(
                {"model_a": model_a, "model_b": model_b, "winner": winner}
            )

            # Count votes for each model
            vote_counts[model_a] += 1
            vote_counts[model_b] += 1

        # Create DataFrame for Elo calculation
        battles_df = pd.DataFrame(battle_data)

        if battles_df.empty:
            return pd.DataFrame()


        # Calculate Elo ratings using Bradley-Terry Model with confidence intervals
        elo_ratings, confidence_intervals = calculate_elo_with_confidence_intervals(
            battles_df, vote_counts
        )

        # Create ranking DataFrame
        ranking_df = create_ranking_dataframe(
            elo_ratings, confidence_intervals, vote_counts
        )

        ranking_data = ranking_df
        ranking_last_updated = datetime.datetime.now()

        return ranking_df
    except Exception as e:
        return pd.DataFrame()


def update_ranking_display():
    """Update ranking display with current data"""
    df = load_ranking_data()
    if df.empty:
        return gr.update(value=df), "**Last Updated:** No data available"

    last_update = (
        ranking_last_updated.strftime("%Y-%m-%d %H:%M:%S")
        if ranking_last_updated
        else "Unknown"
    )
    return gr.update(value=df), f"**Last Updated:** {last_update}"


def force_update_ranking_display():
    """Force update ranking data from HuggingFace (for timer)"""
    df = load_ranking_data(force_reload=True)
    if df.empty:
        return gr.update(value=df), "**Last Updated:** No data available"

    last_update = (
        ranking_last_updated.strftime("%Y-%m-%d %H:%M:%S")
        if ranking_last_updated
        else "Unknown"
    )
    return gr.update(value=df), f"**Last Updated:** {last_update}"


def create_ranking_tab():
    """Create the ranking tab UI component"""
    with gr.Tab("📊 Ranking", id="ranking"):
        gr.Markdown("## 🏆 Model Leaderboard")

        ranking_table = gr.Dataframe(
            headers=[
                "Rank",
                "Model",
                "Score",
                "95% CI (±)",
                "Votes",
                "Organization",
                "License",
            ],
            datatype=[
                "number",
                "str",
                "number",
                "str",
                "number",
                "str",
                "str",
            ],
            label="Model Rankings",
            interactive=False,
            wrap=True,
        )

        ranking_last_update = gr.Markdown("**Last Updated:** Not loaded yet")

        # Timer for auto-refresh every REFRESH_TIME seconds
        ranking_timer = gr.Timer(value=REFRESH_TIME, active=True)

    return ranking_table, ranking_last_update, ranking_timer


def setup_ranking_handlers(demo, ranking_table, ranking_last_update, ranking_timer):
    """Setup event handlers for ranking functionality"""
    
    # Timer tick handler for auto-refresh with force reload
    ranking_timer.tick(
        fn=force_update_ranking_display,
        inputs=[],
        outputs=[ranking_table, ranking_last_update],
    )

    # Auto-load ranking on startup
    demo.load(
        fn=update_ranking_display,
        inputs=[],
        outputs=[ranking_table, ranking_last_update],
    )

    return ranking_table, ranking_last_update
