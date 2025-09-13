"""
Voting module for BigCodeArena
Handles vote submission, data management, and UI components
"""

import gradio as gr
import pandas as pd
import datetime
import os
import threading
from datasets import Dataset, load_dataset


# HuggingFace dataset configuration
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")


def serialize_interactions(interactions):
    """Convert datetime objects in interactions to ISO format strings"""
    if not interactions:
        return interactions
    
    serialized = []
    for interaction in interactions:
        # Handle case where interaction might be a list instead of a dict
        if isinstance(interaction, list):
            # If it's a list, recursively serialize each item
            serialized.append(serialize_interactions(interaction))
        elif isinstance(interaction, dict):
            # If it's a dict, serialize it normally
            serialized_interaction = {}
            for key, value in interaction.items():
                if isinstance(value, datetime.datetime):
                    serialized_interaction[key] = value.isoformat()
                else:
                    serialized_interaction[key] = value
            serialized.append(serialized_interaction)
        else:
            # If it's neither list nor dict, just add it as is
            serialized.append(interaction)
    return serialized


def save_vote_to_hf(
    model_a, model_b, prompt, response_a, response_b, vote_result, interactions_a=None, interactions_b=None, conversation_a=None, conversation_b=None, hf_token=None
):
    """Save vote result to HuggingFace dataset with full conversation history"""
    try:
        # Use global token if not provided
        token = hf_token or HF_TOKEN
        if not token:
            return False, "HuggingFace token not found in environment (HF_TOKEN)"

        if not HF_DATASET_NAME:
            return False, "HuggingFace dataset name not found in environment (HF_DATASET_NAME)"

        # Serialize conversations for JSON compatibility
        serialized_conversation_a = serialize_interactions(conversation_a or [])
        serialized_conversation_b = serialize_interactions(conversation_b or [])

        # Organize interactions by turns - each turn contains a list of interactions
        def organize_interactions_by_turns(interactions, conversation):
            """Organize interactions by conversation turns"""
            if not interactions:
                return []

            # For now, put all interactions in a single turn
            # This can be enhanced later to properly group by conversation turns
            # when we have more context about how interactions are timestamped
            return interactions if interactions else []

        # Organize interactions by turns for both models
        action_a = organize_interactions_by_turns(interactions_a or [], conversation_a or [])
        action_b = organize_interactions_by_turns(interactions_b or [], conversation_b or [])

        # Serialize actions for JSON compatibility
        serialized_action_a = serialize_interactions(action_a)
        serialized_action_b = serialize_interactions(action_b)

        # Create vote data with full conversation history and actions organized by turns
        # Each conversation is a list of messages in format: [{"role": "user"/"assistant", "content": "...", "action": [...]}, ...]
        # Actions are organized as list of lists: [[turn1_interactions], [turn2_interactions], ...]
        vote_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_a": model_a,
            "model_b": model_b,
            "initial_prompt": prompt,  # Convert list to single string
            "action_a": serialized_action_a,  # Actions organized by turns for model A
            "action_b": serialized_action_b,  # Actions organized by turns for model B
            "conversation_a": serialized_conversation_a,  # Full conversation history for model A
            "conversation_b": serialized_conversation_b,  # Full conversation history for model B
            "vote": vote_result,  # "left", "right", "tie", "both_bad"
        }

        # Try to load existing dataset or create new one
        try:
            dataset = load_dataset(HF_DATASET_NAME, split="train", token=token)
            # Convert to pandas DataFrame - handle both Dataset and DatasetDict
            if hasattr(dataset, "to_pandas"):
                df = dataset.to_pandas()
            else:
                df = pd.DataFrame(dataset)
            # Add new vote
            new_df = pd.concat([df, pd.DataFrame([vote_data])], ignore_index=True)
        except Exception as load_error:
            # Create new dataset if it doesn't exist
            new_df = pd.DataFrame([vote_data])

        # Convert back to dataset and push
        new_dataset = Dataset.from_pandas(new_df)
        try:
            new_dataset.push_to_hub(HF_DATASET_NAME, token=token)
            return True, "Vote saved successfully!"
        except Exception as upload_error:
            return False, f"Error uploading to HuggingFace: {str(upload_error)}"
    except Exception as e:
        return False, f"Error saving vote: {str(e)}"


def handle_vote(state0, state1, vote_type):
    """Handle vote submission"""
    if (
        not state0
        or not state1
        or not state0.get("has_output")
        or not state1.get("has_output")
    ):
        return (
            "No output to vote on!",
            gr.update(),
            "**Last Updated:** No data available",
        )

    # Get all user messages and the last responses
    user_messages = []
    response_a = ""
    response_b = ""

    # Collect all user messages from the conversation
    for msg in state0["messages"]:
        if msg["role"] == "user":
            user_messages.append(msg["content"])

    for msg in reversed(state0["messages"]):
        if msg["role"] == "assistant":
            response_a = msg["content"]
            break

    for msg in reversed(state1["messages"]):
        if msg["role"] == "assistant":
            response_b = msg["content"]
            break

    # Get interactions and full conversation history for remote dataset saving
    interactions_a = state0.get("interactions", [])
    interactions_b = state1.get("interactions", [])
    
    # Get full conversation history for both models
    conversation_a = state0.get("messages", [])
    conversation_b = state1.get("messages", [])
    
    # Save vote with full conversation history to remote dataset in background (async)
    def save_vote_background():
        try:
            success, message = save_vote_to_hf(
                state0["model_name"],
                state1["model_name"],
                user_messages[0],
                response_a,
                response_b,
                vote_type,
                interactions_a,
                interactions_b,
                conversation_a,
                conversation_b,
            )

        except Exception as e:
            print(f"Error saving vote: {str(e)}")
            pass
    
    print("Saving vote in background...")
    # Start background upload thread
    upload_thread = threading.Thread(target=save_vote_background)
    upload_thread.daemon = True
    upload_thread.start()
    
    # Return immediately without waiting for upload
    success = True  # Assume success for immediate UI response
    message = "Vote recorded! Uploading data in background..."

    if success:
        # Return immediately without waiting for ranking refresh
        return (
            message + " Clearing conversation...",
            gr.update(),  # Keep existing ranking table
            "**Last Updated:** Processing in background...",
        )
    else:
        return message, gr.update(), "**Last Updated:** Error occurred"


def create_vote_ui():
    """Create vote UI components"""
    # Vote buttons section - only visible after output
    with gr.Row(visible=False) as vote_section:
        gr.Markdown("### 🗳️ Which response is better?")
    
    with gr.Row(visible=False) as vote_buttons_row:
        vote_left_btn = gr.Button(
            "👍 A is Better", variant="primary", size="lg"
        )
        vote_tie_btn = gr.Button(
            "🤝 It's a Tie", variant="secondary", size="lg"
        )
        vote_both_bad_btn = gr.Button(
            "👎 Both are Bad", variant="secondary", size="lg"
        )
        vote_right_btn = gr.Button(
            "👍 B is Better", variant="primary", size="lg"
        )

    # Vote status message
    vote_status = gr.Markdown("", visible=False)
    
    return {
        'vote_section': vote_section,
        'vote_buttons_row': vote_buttons_row,
        'vote_left_btn': vote_left_btn,
        'vote_right_btn': vote_right_btn,
        'vote_tie_btn': vote_tie_btn,
        'vote_both_bad_btn': vote_both_bad_btn,
        'vote_status': vote_status
    }


def should_show_vote_buttons(state0, state1):
    """Check if vote buttons should be shown"""
    return (
        state0
        and state0.get("has_output", False)
        and not state0.get("generating", False)
        and state1
        and state1.get("has_output", False)
        and not state1.get("generating", False)
    )


def get_vote_ui_updates(show_buttons=False):
    """Get UI updates for vote components"""
    return {
        'vote_section': gr.update(visible=show_buttons),
        'vote_buttons_row': gr.update(visible=show_buttons),
        'vote_status': gr.update(visible=False),
        'vote_left_btn': gr.update(interactive=show_buttons),
        'vote_right_btn': gr.update(interactive=show_buttons),
        'vote_tie_btn': gr.update(interactive=show_buttons),
        'vote_both_bad_btn': gr.update(interactive=show_buttons),
    }


def setup_vote_handlers(vote_components, state0_var, state1_var, text_input, ranking_table, ranking_last_update):
    """Setup vote button event handlers"""
    
    def process_vote(state0, state1, vote_type, current_text):
        # Save the vote and get updates
        message, ranking_update, last_update = handle_vote(
            state0, state1, vote_type
        )

        # Show thank you message
        gr.Info(
            "Thank you for your vote! 🎉 Your feedback has been recorded.",
            duration=5,
        )

        # Return only vote status, ranking updates and hide voting interface
        return (
            message,  # vote status message
            gr.update(),  # Keep state0 unchanged
            gr.update(),  # Keep state1 unchanged
            gr.update(),  # Keep chatbot_a unchanged
            gr.update(),  # Keep chatbot_b unchanged
            gr.update(),  # Keep response_a unchanged
            gr.update(),  # Keep response_b unchanged
            gr.update(),  # Keep code_a unchanged
            gr.update(),  # Keep code_b unchanged
            gr.update(),  # Keep sandbox_view_a unchanged
            gr.update(),  # Keep sandbox_view_b unchanged
            gr.update(),  # Keep sandbox_component_a unchanged
            gr.update(),  # Keep sandbox_component_b unchanged
            gr.update(),  # Keep chat_stats_a unchanged
            gr.update(),  # Keep chat_stats_b unchanged
            gr.update(),  # Keep model_display_a unchanged
            gr.update(),  # Keep model_display_b unchanged
            gr.update(visible=False),  # Hide vote_section
            gr.update(visible=False),  # Hide vote_buttons_row
            gr.update(),  # Keep state0_var unchanged
            gr.update(),  # Keep state1_var unchanged
            ranking_update,  # Update ranking_table
            last_update,  # Update ranking_last_update
            gr.update(),  # Keep vote_left_btn unchanged
            gr.update(),  # Keep vote_right_btn unchanged
            gr.update(),  # Keep vote_tie_btn unchanged
            gr.update(),  # Keep vote_both_bad_btn unchanged
            gr.update(),  # Keep text_input unchanged
        )

    # Vote button click handlers
    for vote_btn, vote_type in [
        (vote_components['vote_left_btn'], "left"),
        (vote_components['vote_right_btn'], "right"),
        (vote_components['vote_tie_btn'], "tie"),
        (vote_components['vote_both_bad_btn'], "both_bad"),
    ]:
        vote_btn.click(
            fn=process_vote,
            inputs=[state0_var, state1_var, gr.State(vote_type), text_input],
            outputs=[
                vote_components['vote_status'],  # vote status message
                state0_var,  # state0
                state1_var,  # state1
                # Note: The actual outputs list will need to be filled in by the calling code
                # as it depends on the specific UI components in the main app
            ],
        )
    
    return vote_components
