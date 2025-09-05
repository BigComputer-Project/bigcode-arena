"""
Simple BigCodeArena - A simplified AI coding battle arena
Focuses on core functionality: two models, automatic code extraction, and execution
"""

import gradio as gr
from gradio_sandboxcomponent import SandboxComponent
import pandas as pd
import datetime
import os
import asyncio
import concurrent.futures
import time
from datasets import Dataset, load_dataset
# Import completion utilities
from completion import make_config, registered_api_completion
from sandbox.prompts import GENERAL_SANDBOX_INSTRUCTION
# Import code extraction utilities
from sandbox.code_analyzer import (
    SandboxEnvironment, 
    extract_code_from_markdown, 
)

# Import sandbox execution functions
from sandbox.code_runner import (
    run_html_sandbox,
    run_react_sandbox,
    run_vue_sandbox,
    run_pygame_sandbox,
    run_gradio_sandbox,
    run_streamlit_sandbox,
    run_code_interpreter,
    run_c_code,
    run_cpp_code,
    run_java_code,
    run_golang_code,
    run_rust_code,
    mermaid_to_html,
    javascript_to_html
)

# Import sandbox telemetry
from sandbox.sandbox_telemetry import log_sandbox_telemetry_gradio_fn

# Create a proper sandbox state structure
def create_sandbox_state() -> dict:
    """Create a new sandbox state for a model"""
    return {
        'enable_sandbox': True,
        'enabled_round': 0,
        'sandbox_run_round': 0,
        'edit_round': 0,
        'sandbox_environment': SandboxEnvironment.AUTO,
        'auto_selected_sandbox_environment': None,
        'sandbox_instruction': "Run the extracted code in the appropriate sandbox environment",
        'code_to_execute': "",
        'code_dependencies': ([], []),
        'btn_list_length': 5,
        'sandbox_id': None,
        'chat_session_id': None,
        'conv_id': None,
        "sandbox_output": None,
        "sandbox_error": None,
    }

def reset_sandbox_state(state: dict) -> dict:
    """Reset the sandbox state"""
    state['enabled_round'] = 0
    state['sandbox_run_round'] = 0
    state['edit_round'] = 0
    state['auto_selected_sandbox_environment'] = None
    state['code_to_execute'] = ""
    state['code_dependencies'] = ([], [])
    state['sandbox_error'] = None
    state['sandbox_output'] = None
    state['sandbox_id'] = None
    state['conv_id'] = None
    state['chat_session_id'] = None
    return state

# Load API configuration
def load_api_config():
    """Load API configuration from yaml file"""
    try:
        config = make_config("api_config.yaml")
        return config
    except Exception as e:
        return {}

# Global variables
api_config = load_api_config()
available_models = list(api_config.keys()) if api_config else []

# HuggingFace dataset configuration
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

# Global ranking data cache
ranking_data = None
ranking_last_updated = None

def get_random_models():
    """Get two random models from available models"""
    if len(available_models) < 2:
        return available_models[0] if available_models else None, available_models[0] if available_models else None
    
    import random
    models = random.sample(available_models, 2)
    return models[0], models[1]

def create_chat_state(model_name: str) -> dict:
    """Create a new chat state for a model"""
    return {
        "model_name": model_name,
        "messages": [],
        "sandbox_state": create_sandbox_state(),
        "has_output": False,
        "generating": False,  # Track if model is currently generating
        "interactions": [],  # Store user interactions
    }

def generate_response_with_completion(state, temperature, max_tokens):
    """Generate response using the completion API system with full conversation history"""
    if state is None:
        return state, ""

    # Get the last user message
    user_message = None
    for msg in reversed(state["messages"]):
        if msg["role"] == "user":
            user_message = msg["content"]
            break
    
    if not user_message:
        return state, ""

    # Prepare messages for API call - include full conversation history
    messages = [{"role": "system", "content": GENERAL_SANDBOX_INSTRUCTION}]
    for msg in state["messages"]:
        if msg["role"] in ["user", "assistant"] and msg["content"] is not None:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Get model config
    model_name = state["model_name"]
    if model_name not in api_config:
        return state, f"Error: Model {model_name} not configured"
    
    model_config = api_config[model_name]
    api_type = model_config.get("api_type", "openai")
    
    # retrieve the api completion function from register
    api_completion_func = registered_api_completion[api_type]
    
    # build arguments for api completions
    # Use the actual model identifier from config, not the display name
    actual_model = model_config.get("model", model_name)
    kwargs = {
        "model": actual_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_dict": model_config.get("endpoints", [{}])[0] if model_config.get("endpoints") else None,
        "messages": messages,
    }
    output = api_completion_func(**kwargs)
    
    # Extract the answer from the response
    if isinstance(output, dict) and "answer" in output:
        response_text = output["answer"]
        # Return response as dict with content and interaction keys
        response_dict = {
            "content": response_text,
            "interaction": state.get("interactions", [])
        }
        return state, response_dict
    else:
        error_msg = f"Error: Invalid response format from {api_type}"
        # Return error as dict with content and interaction keys
        error_dict = {
            "content": error_msg,
            "interaction": state.get("interactions", [])
        }
        return state, error_dict


def generate_response_async(state, temperature, max_tokens):
    """Async wrapper for generate_response_with_completion"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Run the synchronous function in a thread pool
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(generate_response_with_completion, state, temperature, max_tokens)
            return future.result()
    finally:
        loop.close()


async def generate_responses_parallel(state0, state1, temperature, max_tokens):
    """Generate responses for both models in parallel with error handling"""
    loop = asyncio.get_event_loop()
    
    # Run both model generations in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future0 = loop.run_in_executor(executor, generate_response_with_completion, state0, temperature, max_tokens)
        future1 = loop.run_in_executor(executor, generate_response_with_completion, state1, temperature, max_tokens)
        
        # Wait for both to complete with error handling
        try:
            result0, result1 = await asyncio.gather(future0, future1, return_exceptions=True)
            
            # Handle exceptions
            if isinstance(result0, Exception):
                result0 = (state0, {"content": f"Error: {str(result0)}", "interaction": []})
            
            if isinstance(result1, Exception):
                result1 = (state1, {"content": f"Error: {str(result1)}", "interaction": []})
                
        except Exception as e:
            # Fallback to sequential processing
            result0 = generate_response_with_completion(state0, temperature, max_tokens)
            result1 = generate_response_with_completion(state1, temperature, max_tokens)
    
    return result0, result1

def extract_and_execute_code(message, sandbox_state):
    """Extract code from message and prepare for execution"""
    if not message:
        return sandbox_state, "", ""
    
    # Extract code using the same logic as code_runner.py
    extract_result = extract_code_from_markdown(
        message=message,
        enable_auto_env=True
    )
    
    if extract_result is None:
        return sandbox_state, "", ""
    
    code, code_language, env_selection, install_command = extract_result

    # Update sandbox state (now a dictionary)
    sandbox_state['code_to_execute'] = code
    sandbox_state['install_command'] = install_command
    sandbox_state['auto_selected_sandbox_environment'] = env_selection
    
    return sandbox_state, code, str(env_selection)

def add_text_and_generate(state0, state1, text, temperature, max_tokens, model_a, model_b):
    """Add text and generate responses for both models"""
    if not text.strip():
        return state0, state1, "", "", "", "", "", "", "", "", "", "", "", ""

    # Initialize states if needed
    if state0 is None or state1 is None:
        if state0 is None:
            state0 = create_chat_state(model_a)
        if state1 is None:
            state1 = create_chat_state(model_b)

    # Add user message to both states
    state0["messages"].append({"role": "user", "content": text})
    state1["messages"].append({"role": "user", "content": text})
    
    # Mark that generation is starting - this will be used to hide vote buttons
    state0["generating"] = True
    state1["generating"] = True

    # Generate responses in parallel
    start_time = time.time()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result0, result1 = loop.run_until_complete(
            generate_responses_parallel(state0, state1, temperature, max_tokens)
        )
        state0, response0 = result0
        state1, response1 = result1
        generation_time = time.time() - start_time
    except Exception as e:
        # Fallback to sequential processing
        state0, response0 = generate_response_with_completion(state0, temperature, max_tokens)
        state1, response1 = generate_response_with_completion(state1, temperature, max_tokens)
        generation_time = time.time() - start_time
    finally:
        loop.close()

    # Add the assistant responses to the message history
    state0["messages"].append({"role": "assistant", "content": response0["content"]})
    state1["messages"].append({"role": "assistant", "content": response1["content"]})

    # Format chat history for display
    chat0 = format_chat_history(state0["messages"])
    chat1 = format_chat_history(state1["messages"])

    # Extract code from responses for sandbox
    sandbox_state0 = (
        state0.get("sandbox_state", create_sandbox_state())
        if state0
        else create_sandbox_state()
    )
    sandbox_state1 = (
        state1.get("sandbox_state", create_sandbox_state())
        if state1
        else create_sandbox_state()
    )

    sandbox_state0, code0, env0 = extract_and_execute_code(response0["content"], sandbox_state0)
    sandbox_state1, code1, env1 = extract_and_execute_code(response1["content"], sandbox_state1)

    # Update sandbox states in the main states
    if state0 is not None:
        state0["sandbox_state"] = sandbox_state0
        state0["has_output"] = True
        state0["generating"] = False  # Mark generation as complete
    if state1 is not None:
        state1["sandbox_state"] = sandbox_state1
        state1["has_output"] = True
        state1["generating"] = False  # Mark generation as complete

    # Clear previous sandbox outputs when new message is sent
    sandbox_output0 = ""
    sandbox_output1 = ""
    # Force clear sandbox components to ensure refresh
    sandbox_component_update0 = gr.update(value=("", False, []), visible=False)
    sandbox_component_update1 = gr.update(value=("", False, []), visible=False)

    # Also clear the sandbox view components to show fresh results
    sandbox_view_a = ""
    sandbox_view_b = ""

    # Run sandbox executions in parallel if both models have code
    if code0.strip() or code1.strip():
        sandbox_start_time = time.time()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Prepare sandbox execution parameters
            install_command0 = sandbox_state0.get('install_command', "") if code0.strip() else ""
            install_command1 = sandbox_state1.get('install_command', "") if code1.strip() else ""
            
            # Run both sandbox executions in parallel
            result0, result1 = loop.run_until_complete(
                run_sandboxes_parallel(
                    sandbox_state0, code0, install_command0,
                    sandbox_state1, code1, install_command1
                )
            )
            sandbox_time = time.time() - sandbox_start_time
            
            # Process results for model A
            if code0.strip():
                sandbox_url0, sandbox_output0, sandbox_error0 = result0

                # Check if this is a web-based environment that should use SandboxComponent
                env_type = sandbox_state0.get('auto_selected_sandbox_environment') or sandbox_state0.get('sandbox_environment')
                # Use the URL directly from the function return
                if sandbox_url0:
                    # Force refresh by using a unique key and clearing first
                    sandbox_component_update0 = gr.update(
                        value=(sandbox_url0, True, []), 
                        visible=True,
                        key=f"sandbox_a_{int(time.time() * 1000)}"  # Unique key to force refresh
                    )

                # Update sandbox view with output and errors
                if sandbox_output0:
                    sandbox_view_a += sandbox_output0
                if sandbox_error0:
                    sandbox_view_a = f"<details closed><summary><strong>🚨 Errors/Warnings</strong></summary>\n\n```\n{sandbox_error0}\n```\n\n</details>\n\n" + sandbox_view_a

            # Process results for model B
            if code1.strip():
                sandbox_url1, sandbox_output1, sandbox_error1 = result1
                # Check if this is a web-based environment that should use SandboxComponent
                env_type = sandbox_state1.get('auto_selected_sandbox_environment') or sandbox_state1.get('sandbox_environment')
                # Use the URL directly from the function return
                if sandbox_url1:
                    # Force refresh by using a unique key and clearing first
                    sandbox_component_update1 = gr.update(
                        value=(sandbox_url1, True, []), 
                        visible=True,
                        key=f"sandbox_b_{int(time.time() * 1000)}"  # Unique key to force refresh
                    )

                if sandbox_output1:
                    sandbox_view_b += sandbox_output1
                if sandbox_error1:
                    sandbox_view_b = f"<details closed><summary><strong>🚨 Errors/Warnings</strong></summary>\n\n```\n{sandbox_error1}\n```\n\n</details>\n\n" + sandbox_view_b
                    
        except Exception as e:
            # Fallback to sequential processing
            if code0.strip():
                install_command0 = sandbox_state0.get('install_command', "")
                sandbox_url0, sandbox_output0, sandbox_error0 = run_sandbox_code(sandbox_state0, code0, install_command0)
                if sandbox_url0:
                    sandbox_component_update0 = gr.update(
                        value=(sandbox_url0, True, []), 
                        visible=True,
                        key=f"sandbox_a_fallback_{int(time.time() * 1000)}"
                    )
                if sandbox_output0:
                    sandbox_view_a += sandbox_output0
                if sandbox_error0:
                    sandbox_view_a = f"<details closed><summary><strong>🚨 Errors/Warnings</strong></summary>\n\n```\n{sandbox_error0}\n```\n\n</details>\n\n" + sandbox_view_a
            
            if code1.strip():
                install_command1 = sandbox_state1.get('install_command', "")
                sandbox_url1, sandbox_output1, sandbox_error1 = run_sandbox_code(sandbox_state1, code1, install_command1)
                if sandbox_url1:
                    sandbox_component_update1 = gr.update(
                        value=(sandbox_url1, True, []), 
                        visible=True,
                        key=f"sandbox_b_fallback_{int(time.time() * 1000)}"
                    )
                if sandbox_output1:
                    sandbox_view_b += f"## Output\n{sandbox_output1}"
                if sandbox_error1:
                    sandbox_view_b = f"<details closed><summary><strong>🚨 Errors/Warnings</strong></summary>\n\n```\n{sandbox_error1}\n```\n\n</details>\n\n" + sandbox_view_b
            
            sandbox_time = time.time() - sandbox_start_time
        finally:
            loop.close()
    else:
        # No code to execute, but still ensure sandbox components are cleared
        sandbox_component_update0 = gr.update(value=("", False, []), visible=False)
        sandbox_component_update1 = gr.update(value=("", False, []), visible=False)

    # Calculate conversation statistics
    turn_count_a = (
        len(
            [
                msg
                for msg in state0["messages"]
                if msg["role"] == "assistant" and msg["content"]
            ]
        )
        if state0
        else 0
    )
    turn_count_b = (
        len(
            [
                msg
                for msg in state1["messages"]
                if msg["role"] == "assistant" and msg["content"]
            ]
        )
        if state1
        else 0
    )

    # Format conversation statistics
    chat_stats_a = f"**Conversation:** {turn_count_a} turns | **Total Messages:** {len(state0['messages']) if state0 else 0}"
    chat_stats_b = f"**Conversation:** {turn_count_b} turns | **Total Messages:** {len(state1['messages']) if state1 else 0}"

    return state0, state1, chat0, chat1, response0, response1, code0, code1, env0, env1, sandbox_state0, sandbox_state1, sandbox_output0, sandbox_output1, sandbox_component_update0, sandbox_component_update1, chat_stats_a, chat_stats_b, sandbox_view_a, sandbox_view_b

def format_chat_history(messages):
    """Format messages for chat display with turn numbers"""
    formatted = []
    
    for msg in messages:
        if msg["role"] == "user" and msg["content"]:
            # Add turn number to user messages
            formatted.append({
                "role": "user", 
                "content": msg['content']
            })
        elif msg["role"] == "assistant" and msg["content"]:
            # Add turn number to assistant messages
            formatted.append({
                "role": "assistant", 
                "content": msg['content']
            })
    
    return formatted

def clear_chat(state0, state1):
    """Clear chat history"""
    if state0 and "sandbox_state" in state0:
        reset_sandbox_state(state0["sandbox_state"])
        state0["interactions"] = []  # Clear interactions
        state0["generating"] = False  # Reset generating flag
    if state1 and "sandbox_state" in state1:
        reset_sandbox_state(state1["sandbox_state"])
        state1["interactions"] = []  # Clear interactions
        state1["generating"] = False  # Reset generating flag

    # Get current model names for display
    model_a, model_b = get_random_models()

    return (
        None,  # state0
        None,  # state1
        "",    # chatbot_a
        "",    # chatbot_b
        "",    # response_a
        "",    # response_b
        "",    # code_a
        "",    # code_b
        None,  # sandbox_state0
        None,  # sandbox_state1
        "",    # sandbox_view_a
        "",    # sandbox_view_b
        gr.update(value=("", False, []), visible=False),  # sandbox_component_a
        gr.update(value=("", False, []), visible=False),  # sandbox_component_b
        "**Conversation:** 0 turns | **Total Messages:** 0",  # chat_stats_a
        "**Conversation:** 0 turns | **Total Messages:** 0",  # chat_stats_b
        "",    # sandbox_view_a (duplicate)
        "",    # sandbox_view_b (duplicate)
        f"**Model A:** {model_a}",  # model_display_a
        f"**Model B:** {model_b}",  # model_display_b
        "",    # text_input
        gr.update(visible=False),  # vote_section
        gr.update(visible=False),  # vote_buttons_row
        "",    # vote_status
        gr.update(interactive=False),  # vote_left_btn
        gr.update(interactive=False),  # vote_right_btn
        gr.update(interactive=False),  # vote_tie_btn
        gr.update(interactive=False),  # vote_both_bad_btn
    )

def retry_last_message(state0, state1, model_a, model_b):
    """Retry the last user message"""
    if not state0 or not state1:
        return state0, state1, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
    
    # Get the last user message
    last_user_message = ""
    for msg in reversed(state0["messages"]):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break
    
    if not last_user_message:
        return state0, state1, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
    
    # Remove the last user message and assistant responses from both states
    if state0["messages"] and state0["messages"][-1]["role"] == "assistant":
        state0["messages"].pop()  # Remove last assistant response
    if state0["messages"] and state0["messages"][-1]["role"] == "user":
        state0["messages"].pop()  # Remove last user message
    
    if state1["messages"] and state1["messages"][-1]["role"] == "assistant":
        state1["messages"].pop()  # Remove last assistant response
    if state1["messages"] and state1["messages"][-1]["role"] == "user":
        state1["messages"].pop()  # Remove last user message
    
    # Generate new responses with the same message
    result = add_text_and_generate(state0, state1, last_user_message, 0.4, 8192, model_a, model_b)
    
    # Extract the state from the result
    new_state0, new_state1 = result[0], result[1]
    
    # Check if both models have output and are not generating to show vote buttons
    show_vote_buttons = (
        new_state0
        and new_state0.get("has_output", False)
        and not new_state0.get("generating", False)
        and new_state1
        and new_state1.get("has_output", False)
        and not new_state1.get("generating", False)
    )
    
    # Return all the original outputs plus the updated state for run buttons
    return (
        new_state0,  # state0
        new_state1,  # state1
        result[2],  # chatbot_a (chat0)
        result[3],  # chatbot_b (chat1)
        result[4]["content"] if isinstance(result[4], dict) else result[4],  # response_a (response0)
        result[5]["content"] if isinstance(result[5], dict) else result[5],  # response_b (response1)
        result[6],  # code_a (code0)
        result[7],  # code_b (code1)
        result[10] if len(result) > 10 else "",  # sandbox_state0
        result[11] if len(result) > 11 else "",  # sandbox_state1
        result[12] if len(result) > 12 else "",  # sandbox_output0
        result[13] if len(result) > 13 else "",  # sandbox_output1
        (
            result[14] if len(result) > 14 else gr.update(visible=False)
        ),  # sandbox_component_update0
        (
            result[15] if len(result) > 15 else gr.update(visible=False)
        ),  # sandbox_component_update1
        (
            result[16] if len(result) > 16 else "**Conversation:** 0 turns"
        ),  # chat_stats_a
        (
            result[17] if len(result) > 17 else "**Conversation:** 0 turns"
        ),  # chat_stats_b
        result[18] if len(result) > 18 else "",  # sandbox_view_a
        result[19] if len(result) > 19 else "",  # sandbox_view_b
        new_state0,  # state0_var
        new_state1,  # state1_var
        last_user_message,  # Keep original text input
        f"**Model A:** {model_a}",  # Update model display A
        f"**Model B:** {model_b}",  # Update model display B
        gr.update(visible=show_vote_buttons),  # vote_section
        gr.update(visible=show_vote_buttons),  # vote_buttons_row
        gr.update(visible=False),  # vote_status
        gr.update(interactive=show_vote_buttons),  # vote_left_btn
        gr.update(interactive=show_vote_buttons),  # vote_right_btn
        gr.update(interactive=show_vote_buttons),  # vote_tie_btn
        gr.update(interactive=show_vote_buttons),  # vote_both_bad_btn
    )

def send_to_left_only(state0, state1, text, temperature, max_tokens, model_a, model_b):
    """Send message to left model (Model A) only"""
    if not text.strip():
        return state0, state1, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
    
    # Initialize states if needed
    if state0 is None:
        state0 = create_chat_state(model_a)
    if state1 is None:
        state1 = create_chat_state(model_b)
    
    # Add user message to left state only
    state0["messages"].append({"role": "user", "content": text})
    state0["generating"] = True
    
    # Generate response for left model only
    state0, response0 = generate_response_with_completion(state0, temperature, max_tokens)
    state0["messages"].append({"role": "assistant", "content": response0["content"]})
    state0["has_output"] = True
    state0["generating"] = False
    
    # Format chat history for display
    chat0 = format_chat_history(state0["messages"])
    chat1 = format_chat_history(state1["messages"]) if state1 else []
    
    # Extract code from response for sandbox
    sandbox_state0 = state0.get("sandbox_state", create_sandbox_state())
    sandbox_state0, code0, env0 = extract_and_execute_code(response0["content"], sandbox_state0)
    state0["sandbox_state"] = sandbox_state0
    
    # Clear previous sandbox outputs
    sandbox_output0 = ""
    sandbox_component_update0 = gr.update(value=("", False, []), visible=False)
    sandbox_view_a = ""
    
    # Run sandbox execution if there's code
    if code0.strip():
        install_command0 = sandbox_state0.get('install_command', "")
        sandbox_url0, sandbox_output0, sandbox_error0 = run_sandbox_code(sandbox_state0, code0, install_command0)
        if sandbox_url0:
            sandbox_component_update0 = gr.update(
                value=(sandbox_url0, True, []), 
                visible=True,
                key=f"sandbox_a_{int(time.time() * 1000)}"
            )
        if sandbox_output0:
            sandbox_view_a += f"# Output\n{sandbox_output0}"
        if sandbox_error0:
            sandbox_view_a = f"<details closed><summary><strong>🚨 Errors/Warnings</strong></summary>\n\n```\n{sandbox_error0.strip()}\n```\n\n</details>\n\n" + sandbox_view_a
    
    # Calculate conversation statistics
    turn_count_a = len([msg for msg in state0["messages"] if msg["role"] == "assistant" and msg["content"]])
    turn_count_b = len([msg for msg in state1["messages"] if msg["role"] == "assistant" and msg["content"]]) if state1 else 0
    
    chat_stats_a = f"**Conversation:** {turn_count_a} turns | **Total Messages:** {len(state0['messages'])}"
    chat_stats_b = f"**Conversation:** {turn_count_b} turns | **Total Messages:** {len(state1['messages']) if state1 else 0}"
    
    # Don't show vote buttons since only one model responded
    show_vote_buttons = False
    
    return (
        state0,  # state0
        state1,  # state1
        chat0,  # chatbot_a
        chat1,  # chatbot_b
        response0["content"] if isinstance(response0, dict) else response0,  # response_a
        "",  # response_b (empty)
        code0,  # code_a
        "",  # code_b (empty)
        sandbox_state0,  # sandbox_state0
        state1.get("sandbox_state", create_sandbox_state()) if state1 else create_sandbox_state(),  # sandbox_state1
        sandbox_output0,  # sandbox_output0
        "",  # sandbox_output1 (empty)
        sandbox_component_update0,  # sandbox_component_update0
        gr.update(value=("", False, []), visible=False),  # sandbox_component_update1
        chat_stats_a,  # chat_stats_a
        chat_stats_b,  # chat_stats_b
        sandbox_view_a,  # sandbox_view_a
        "",  # sandbox_view_b (empty)
        state0,  # state0_var
        state1,  # state1_var
        text,  # Keep original text input
        f"**Model A:** {model_a}",  # Update model display A
        f"**Model B:** {model_b}",  # Update model display B
        gr.update(visible=show_vote_buttons),  # vote_section
        gr.update(visible=show_vote_buttons),  # vote_buttons_row
        gr.update(visible=False),  # vote_status
        gr.update(interactive=show_vote_buttons),  # vote_left_btn
        gr.update(interactive=show_vote_buttons),  # vote_right_btn
        gr.update(interactive=show_vote_buttons),  # vote_tie_btn
        gr.update(interactive=show_vote_buttons),  # vote_both_bad_btn
    )

def send_to_right_only(state0, state1, text, temperature, max_tokens, model_a, model_b):
    """Send message to right model (Model B) only"""
    if not text.strip():
        return state0, state1, "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
    
    # Initialize states if needed
    if state0 is None:
        state0 = create_chat_state(model_a)
    if state1 is None:
        state1 = create_chat_state(model_b)
    
    # Add user message to right state only
    state1["messages"].append({"role": "user", "content": text})
    state1["generating"] = True
    
    # Generate response for right model only
    state1, response1 = generate_response_with_completion(state1, temperature, max_tokens)
    state1["messages"].append({"role": "assistant", "content": response1["content"]})
    state1["has_output"] = True
    state1["generating"] = False
    
    # Format chat history for display
    chat0 = format_chat_history(state0["messages"]) if state0 else []
    chat1 = format_chat_history(state1["messages"])
    
    # Extract code from response for sandbox
    sandbox_state1 = state1.get("sandbox_state", create_sandbox_state())
    sandbox_state1, code1, env1 = extract_and_execute_code(response1["content"], sandbox_state1)
    state1["sandbox_state"] = sandbox_state1
    
    # Clear previous sandbox outputs
    sandbox_output1 = ""
    sandbox_component_update1 = gr.update(value=("", False, []), visible=False)
    sandbox_view_b = ""
    
    # Run sandbox execution if there's code
    if code1.strip():
        install_command1 = sandbox_state1.get('install_command', "")
        sandbox_url1, sandbox_output1, sandbox_error1 = run_sandbox_code(sandbox_state1, code1, install_command1)
        if sandbox_url1:
            sandbox_component_update1 = gr.update(
                value=(sandbox_url1, True, []), 
                visible=True,
                key=f"sandbox_b_{int(time.time() * 1000)}"
            )
        if sandbox_output1:
            sandbox_view_b += f"# Output\n{sandbox_output1}"
        if sandbox_error1:
            sandbox_view_b = f"<details closed><summary><strong>🚨 Errors/Warnings</strong></summary>\n\n```\n{sandbox_error1.strip()}\n```\n\n</details>\n\n" + sandbox_view_b
    
    # Calculate conversation statistics
    turn_count_a = len([msg for msg in state0["messages"] if msg["role"] == "assistant" and msg["content"]]) if state0 else 0
    turn_count_b = len([msg for msg in state1["messages"] if msg["role"] == "assistant" and msg["content"]])
    
    chat_stats_a = f"**Conversation:** {turn_count_a} turns | **Total Messages:** {len(state0['messages']) if state0 else 0}"
    chat_stats_b = f"**Conversation:** {turn_count_b} turns | **Total Messages:** {len(state1['messages'])}"
    
    # Don't show vote buttons since only one model responded
    show_vote_buttons = False
    
    return (
        state0,  # state0
        state1,  # state1
        chat0,  # chatbot_a
        chat1,  # chatbot_b
        "",  # response_a (empty)
        response1["content"] if isinstance(response1, dict) else response1,  # response_b
        "",  # code_a (empty)
        code1,  # code_b
        state0.get("sandbox_state", create_sandbox_state()) if state0 else create_sandbox_state(),  # sandbox_state0
        sandbox_state1,  # sandbox_state1
        "",  # sandbox_output0 (empty)
        sandbox_output1,  # sandbox_output1
        gr.update(value=("", False, []), visible=False),  # sandbox_component_update0
        sandbox_component_update1,  # sandbox_component_update1
        chat_stats_a,  # chat_stats_a
        chat_stats_b,  # chat_stats_b
        "",  # sandbox_view_a (empty)
        sandbox_view_b,  # sandbox_view_b
        state0,  # state0_var
        state1,  # state1_var
        text,  # Keep original text input
        f"**Model A:** {model_a}",  # Update model display A
        f"**Model B:** {model_b}",  # Update model display B
        gr.update(visible=show_vote_buttons),  # vote_section
        gr.update(visible=show_vote_buttons),  # vote_buttons_row
        gr.update(visible=False),  # vote_status
        gr.update(interactive=show_vote_buttons),  # vote_left_btn
        gr.update(interactive=show_vote_buttons),  # vote_right_btn
        gr.update(interactive=show_vote_buttons),  # vote_tie_btn
        gr.update(interactive=show_vote_buttons),  # vote_both_bad_btn
    )


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
    import threading
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


def run_sandbox_code(sandbox_state: dict, code: str, install_command: str) -> tuple[str, str, str]:
    """Run code in the appropriate sandbox environment"""
    if not code.strip():
        return "", "", "No code to run"

    # Update sandbox state
    sandbox_state['code_to_execute'] = code
    sandbox_state['install_command'] = install_command

    # Determine environment
    env = sandbox_state.get('auto_selected_sandbox_environment') or sandbox_state.get('sandbox_environment')
    print(f"DEBUG: env: {env}")
    try:
        if env == SandboxEnvironment.HTML:
            sandbox_url, sandbox_id, stderr = run_html_sandbox(code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr

        elif env == SandboxEnvironment.REACT:
            result = run_react_sandbox(code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = result['sandbox_id']
            return result['sandbox_url'], "", result['stderr']

        elif env == SandboxEnvironment.VUE:
            result = run_vue_sandbox(code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = result['sandbox_id']
            return result['sandbox_url'], "", result['stderr']

        elif env == SandboxEnvironment.PYGAME:
            result = run_pygame_sandbox(code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = result['sandbox_id']
            return result['sandbox_url'], "", result['stderr']

        elif env == SandboxEnvironment.GRADIO:
            print(f"DEBUG: running gradio sandbox")
            sandbox_url, sandbox_id, stderr = run_gradio_sandbox(code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr

        elif env == SandboxEnvironment.STREAMLIT:
            sandbox_url, sandbox_id, stderr = run_streamlit_sandbox(code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr

        elif env == SandboxEnvironment.MERMAID:
            # Convert Mermaid to HTML and run in HTML sandbox
            html_code = mermaid_to_html(code, theme='light')
            sandbox_url, sandbox_id, stderr = run_html_sandbox(html_code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr

        elif env == SandboxEnvironment.PYTHON_RUNNER:
            print(f"DEBUG: running python runner")
            output, stderr = run_code_interpreter(code, 'python', install_command)
            return "", output, stderr

        elif env == SandboxEnvironment.JAVASCRIPT_RUNNER:
            html_code = javascript_to_html(code)
            sandbox_url, sandbox_id, stderr = run_html_sandbox(html_code, install_command, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr

        elif env == SandboxEnvironment.C_RUNNER:
            output, stderr = run_c_code(code, sandbox_state.get('sandbox_id'))
            return "", output, stderr

        elif env == SandboxEnvironment.CPP_RUNNER:
            output, stderr = run_cpp_code(code, sandbox_state.get('sandbox_id'))
            return "", output, stderr

        elif env == SandboxEnvironment.JAVA_RUNNER:
            output, stderr = run_java_code(code, sandbox_state.get('sandbox_id'))
            return "", output, stderr

        elif env == SandboxEnvironment.GOLANG_RUNNER:
            output, stderr = run_golang_code(code, sandbox_state.get('sandbox_id'))
            return "", output, stderr

        elif env == SandboxEnvironment.RUST_RUNNER:
            output, stderr = run_rust_code(code, sandbox_state.get('sandbox_id'))
            return "", output, stderr

        else:
            # Fallback to Python runner
            output, stderr = run_code_interpreter(code, 'python', install_command)
            return "", output, stderr

    except Exception as e:
        return "", "", str(e)


async def run_sandbox_code_async(sandbox_state: dict, code: str, install_command: str) -> tuple[str, str, str]:
    """Async wrapper for run_sandbox_code"""
    loop = asyncio.get_event_loop()
    
    # Run sandbox execution in a thread pool to avoid blocking
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = loop.run_in_executor(executor, run_sandbox_code, sandbox_state, code, install_command)
        return await future


async def run_sandboxes_parallel(sandbox_state0, code0, install_command0, sandbox_state1, code1, install_command1):
    """Run both sandbox executions in parallel with error handling"""
    loop = asyncio.get_event_loop()
    
    # Create tasks for both sandbox executions
    task0 = loop.run_in_executor(None, run_sandbox_code, sandbox_state0, code0, install_command0)
    task1 = loop.run_in_executor(None, run_sandbox_code, sandbox_state1, code1, install_command1)
    
    # Wait for both to complete with error handling
    try:
        result0, result1 = await asyncio.gather(task0, task1, return_exceptions=True)
        
        # Handle exceptions
        if isinstance(result0, Exception):
            result0 = ("", "", f"Sandbox execution error: {str(result0)}")
        
        if isinstance(result1, Exception):
            result1 = ("", "", f"Sandbox execution error: {str(result1)}")
            
    except Exception as e:
        # Fallback to sequential processing
        result0 = run_sandbox_code(sandbox_state0, code0, install_command0)
        result1 = run_sandbox_code(sandbox_state1, code1, install_command1)
    
    return result0, result1


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


def load_ranking_data(hf_token=None, force_reload=False):
    """Load and calculate ranking data from HuggingFace dataset"""
    global ranking_data, ranking_last_updated

    try:
        # Use global token if not provided
        token = hf_token or HF_TOKEN
        if not token:
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

        # Calculate rankings
        model_stats = {}

        for _, row in df.iterrows():
            model_a = row["model_a"]
            model_b = row["model_b"]
            vote = row["vote"]

            # Initialize models if not exists
            if model_a not in model_stats:
                model_stats[model_a] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}
            if model_b not in model_stats:
                model_stats[model_b] = {"wins": 0, "losses": 0, "ties": 0, "total": 0}

            # Update stats based on vote
            if vote == "left":  # Model A wins
                model_stats[model_a]["wins"] += 1
                model_stats[model_b]["losses"] += 1
            elif vote == "right":  # Model B wins
                model_stats[model_b]["wins"] += 1
                model_stats[model_a]["losses"] += 1
            elif vote == "tie":
                model_stats[model_a]["ties"] += 1
                model_stats[model_b]["ties"] += 1
            # both_bad doesn't count as win/loss for either

            model_stats[model_a]["total"] += 1
            model_stats[model_b]["total"] += 1

        # Convert to DataFrame and calculate win rate
        ranking_list = []
        for model, stats in model_stats.items():
            win_rate = (
                (stats["wins"] + stats["ties"]) / max(stats["total"], 1) * 100
            )
            ranking_list.append(
                {
                    "Model": model,
                    "Win Rate (%)": round(win_rate, 1),
                    "Wins": stats["wins"],
                    "Losses": stats["losses"],
                    "Ties": stats["ties"],
                    "Total Battles": stats["total"],
                }
            )

        # Sort by win rate
        ranking_df = pd.DataFrame(ranking_list).sort_values(
            "Win Rate (%)", ascending=False
        )
        ranking_df["Rank"] = range(1, len(ranking_df) + 1)

        # Reorder columns
        ranking_df = ranking_df[
            ["Rank", "Model", "Win Rate (%)", "Wins", "Losses", "Ties", "Total Battles"]
        ]

        ranking_data = ranking_df
        ranking_last_updated = datetime.datetime.now()

        return ranking_df
    except Exception as e:
        return pd.DataFrame()


def instantiate_send_button():
    """Create a send button with icon"""
    return gr.Button(
        "🚀",
        size="lg",
        scale=0,
        min_width=60,
        variant="primary",
        elem_id="send-btn"
    )

def instantiate_retry_button():
    """Create a retry button with icon"""
    return gr.Button(
        "🔄",
        size="lg",
        scale=0,
        min_width=60,
        variant="secondary",
        elem_id="retry-btn"
    )

def instantiate_send_left_button():
    """Create a send left button with icon"""
    return gr.Button(
        "⬅️",
        size="lg",
        scale=0,
        min_width=60,
        variant="secondary",
        elem_id="send-left-btn"
    )

def instantiate_send_right_button():
    """Create a send right button with icon"""
    return gr.Button(
        "➡️",
        size="lg",
        scale=0,
        min_width=60,
        variant="secondary",
        elem_id="send-right-btn"
    )

def instantiate_clear_button():
    """Create a clear button with icon"""
    return gr.Button(
        "🗑️",
        size="sm",
        scale=0,
        min_width=40,
        variant="secondary",
        elem_id="clear-btn"
    )

def build_ui():
    """Build a UI for the coding arena with integrated sandbox"""

    # Get random models for this session
    model_a, model_b = get_random_models()

    with gr.Blocks(title="BigCodeArena", theme=gr.themes.Soft()) as demo:
        # Add custom CSS for centering and button styling
        demo.css = """
        .center-text {
            text-align: center !important;
        }
        .input-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .input-row .gr-textbox {
            flex: 1;
        }
        .input-row .gr-button {
            flex-shrink: 0;
            height: 40px;
            font-size: 16px;
        }
        .button-grid {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .button-grid .gr-row {
            display: flex;
            gap: 8px;
        }
        .button-grid .gr-button {
            flex: 1;
            min-width: 60px;
        }
        """
        
        gr.Markdown("# 🌸 BigCodeArena - Start Your Vibe Coding!", elem_classes="center-text")

        # Main tabs
        with gr.Tabs():
            # Arena Tab
            with gr.Tab("🥊 Arena", id="arena"):

                # Model display (non-interactive)
                with gr.Row():
                    with gr.Column():
                        model_display_a = gr.Markdown(
                            f"**Model A:** {model_a}", visible=False
                        )
                    with gr.Column():
                        model_display_b = gr.Markdown(
                            f"**Model B:** {model_b}", visible=False
                        )

                # Sandbox section with tabs for each model - Collapsible and open by default
                with gr.Accordion("🏗️ Code Execution & Sandbox", open=True):

                    with gr.Row():
                        # Model A Sandbox
                        with gr.Column():
                            gr.Markdown("### Model A Sandbox")
                            with gr.Tabs():
                                with gr.Tab("View"):
                                    sandbox_view_a = gr.Markdown(
                                        "**Sandbox output will appear here automatically**"
                                    )
                                    sandbox_component_a = SandboxComponent(
                                        value=("", False, []),
                                        label="Model A Sandbox",
                                        visible=False,
                                    )
                                with gr.Tab("Code"):
                                    code_a = gr.Code(
                                        label="Extracted Code",
                                        language="python",
                                        lines=8,
                                        interactive=False,
                                    )

                        # Model B Sandbox
                        with gr.Column():
                            gr.Markdown("### Model B Sandbox")
                            with gr.Tabs():
                                with gr.Tab("View"):
                                    sandbox_view_b = gr.Markdown(
                                        "**Sandbox output will appear here automatically**"
                                    )
                                    sandbox_component_b = SandboxComponent(
                                        value=("", False, []),
                                        label="Model B Sandbox",
                                        visible=False,
                                    )
                                with gr.Tab("Code"):
                                    code_b = gr.Code(
                                        label="Extracted Code",
                                        language="python",
                                        lines=8,
                                        interactive=False,
                                    )

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

                # Main chat interface - Collapsible and hidden by default
                with gr.Accordion("💬 Chat Interface", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Model A")
                            chatbot_a = gr.Chatbot(
                                label="Model A",
                                height=300,
                                show_copy_button=True,
                                type="messages",
                            )
                            chat_stats_a = gr.Markdown("**Conversation:** 0 turns")

                        with gr.Column():
                            gr.Markdown("## Model B")
                            chatbot_b = gr.Chatbot(
                                label="Model B",
                                height=300,
                                show_copy_button=True,
                                type="messages",
                            )
                            chat_stats_b = gr.Markdown("**Conversation:** 0 turns")

                # Input section with 2x2 button grid
                with gr.Row(elem_classes="input-row"):
                    text_input = gr.Textbox(
                        label="Enter your coding prompt",
                        placeholder="e.g., 'Write a Python function to calculate fibonacci numbers'",
                        lines=1,
                        scale=1
                    )
                    with gr.Column(scale=0, min_width=140, elem_classes="button-grid"):
                        with gr.Row():
                            send_btn = instantiate_send_button()
                            retry_btn = instantiate_retry_button()
                        with gr.Row():
                            send_left_btn = instantiate_send_left_button()
                            send_right_btn = instantiate_send_right_button()
                            
                # Additional control buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    refresh_models_btn = gr.Button(
                        "🔄 New Random Models", variant="secondary"
                    )

                # Advanced Settings (Collapsible)
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.4,
                                step=0.1,
                                label="Temperature",
                            )
                        with gr.Column(scale=1):
                            max_tokens = gr.Slider(
                                minimum=1024,
                                maximum=32768,
                                value=8192,
                                label="Max Tokens",
                            )
                # Examples
                gr.Examples(
                    examples=[
                        [
                            "使用SVG绘制春节主题的动态图案，包括：1）一个红色的灯笼，带有金色的流苏 2）一个金色的福字，使用书法字体 3）背景添加一些烟花效果 4）在灯笼和福字周围添加一些祥云图案。确保图案布局美观，颜色搭配符合春节传统风格。"
                        ],
                        [
                            "SVGを使用して日本の伝統的な和柄パターンを描画してください。1）波紋（さざなみ）模様 2）市松模様 3）麻の葉模様 4）雷文（らいもん）模様を含めてください。色は伝統的な日本の色（藍色、朱色、金色など）を使用し、レイアウトはバランスよく配置してください。"
                        ],
                        [
                            "Write HTML with P5.js that simulates 25 particles in a vacuum space of a cylindrical container, bouncing within its boundaries. Use different colors for each ball and ensure they leave a trail showing their movement. Add a slow rotation of the container to give better view of what's going on in the scene. Make sure to create proper collision detection and physic rules to ensure particles remain in the container. Add an external spherical container. Add a slow zoom in and zoom out effect to the whole scene."
                        ],
                        [
                            "Write a Python script to scrape NVIDIA's stock price for the past month using the yfinance library. Clean the data and create an interactive visualization using Matplotlib. Include: 1) A candlestick chart showing daily price movements 2) A line chart with 7-day and 30-day moving averages. Add hover tooltips showing exact values and date. Make the layout professional with proper titles and axis labels."
                        ],
                        [
                            "Write a Python script that uses the Gradio library to create a functional calculator. The calculator should support basic arithmetic operations: addition, subtraction, multiplication, and division. It should have two input fields for numbers and a dropdown menu to select the operation."
                        ],
                        [
                            "Write a Todo list app using React.js. The app should allow users to add, delete, and mark tasks as completed. Include features like filtering tasks by status (completed, active), sorting tasks by priority, and displaying the total number of tasks."
                        ],
                        [
                            "Write a Python script using the Streamlit library to create a web application for uploading and displaying files. The app should allow users to upload files of type .csv or .txt. If a .csv file is uploaded, display its contents as a table using Streamlit's st.dataframe() method. If a .txt file is uploaded, display its content as plain text."
                        ],
                        [
                            "Write a Python function to solve the Trapping Rain Water problem. The function should take a list of non-negative integers representing the height of bars in a histogram and return the total amount of water trapped between the bars after raining. Use an efficient algorithm with a time complexity of O(n)."
                        ],
                        [
                            "Create a simple Pygame script for a game where the player controls a bouncing ball that changes direction when it collides with the edges of the window. Add functionality for the player to control a paddle using arrow keys, aiming to keep the ball from touching the bottom of the screen. Include basic collision detection and a scoring system that increases as the ball bounces off the paddle. You need to add clickable buttons to start the game, and reset the game."
                        ],
                        [
                            "Create a financial management Dashboard using Vue.js, focusing on local data handling without APIs. Include features like a clean dashboard for tracking income and expenses, dynamic charts for visualizing finances, and a budget planner. Implement functionalities for adding, editing, and deleting transactions, as well as filtering by date or category. Ensure responsive design and smooth user interaction for an intuitive experience."
                        ],
                        [
                            "Create a Mermaid diagram to visualize a flowchart of a user login process. Include the following steps: User enters login credentials; Credentials are validated; If valid, the user is directed to the dashboard; If invalid, an error message is shown, and the user can retry or reset the password."
                        ],
                        [
                            "Write a Python function to calculate the Fibonacci sequence up to n numbers. Then write test cases to verify the function works correctly for edge cases like negative numbers, zero, and large inputs."
                        ],
                        [
                            "Build an HTML page for a Kanban board with three columns with Vue.js: To Do, In Progress, and Done. Each column should allow adding, moving, and deleting tasks. Implement drag-and-drop functionality using Vue Draggable and persist the state using Vuex."
                        ],
                        [
                            "Develop a Streamlit app that takes a CSV file as input and provides: 1) Basic statistics about the data 2) Interactive visualizations using Plotly 3) A data cleaning interface with options to handle missing values 4) An option to download the cleaned data."
                        ],
                        [
                            "Write an HTML page with embedded JavaScript that creates an interactive periodic table. Each element should display its properties on hover and allow filtering by category (metals, non-metals, etc.). Include a search bar to find elements by name or symbol."
                        ],
                        [
                            "Here's a Python function that sorts a list of dictionaries by a specified key:\n\n```python\ndef sort_dicts(data, key):\n    return sorted(data, key=lambda x: x[key])\n```\n\nWrite test cases to verify the function works correctly for edge cases like empty lists, missing keys, and different data types. If you use unittest, please use `unittest.main(argv=['first-arg-is-ignored'], exit=False)` to run the tests."
                        ],
                        [
                            "Create a React component for a fitness tracker that shows: 1) Daily step count 2) Calories burned 3) Distance walked 4) A progress bar for daily goals."
                        ],
                        [
                            "Build a Vue.js dashboard for monitoring server health. Include: 1) Real-time CPU and memory usage graphs 2) Disk space visualization 3) Network activity monitor 4) Alerts for critical thresholds."
                        ],
                        [
                            "Write a C program that calculates and prints the first 100 prime numbers in a formatted table with 10 numbers per row. Include a function to check if a number is prime and use it in your solution."
                        ],
                        [
                            "Write a C++ program that implements a simple calculator using object-oriented programming. Create a Calculator class with methods for addition, subtraction, multiplication, and division. Include error handling for division by zero."
                        ],
                        [
                            "Write a Rust program that generates and prints a Pascal's Triangle with 10 rows. Format the output to center-align the numbers in each row."
                        ],
                        [
                            "Write a Java program that simulates a simple bank account system. Create a BankAccount class with methods for deposit, withdrawal, and balance inquiry. Include error handling for insufficient funds and demonstrate its usage with a few transactions."
                        ],
                        [
                            "Write a Go program that calculates and prints the Fibonacci sequence up to the 50th number. Format the output in a table with 5 numbers per row and include the index of each Fibonacci number."
                        ],
                        [
                            "Write a C program that calculates and prints a histogram of letter frequencies from a predefined string. Use ASCII art to display the histogram vertically."
                        ],
                        [
                            "Write a C++ program that implements a simple stack data structure with push, pop, and peek operations. Demonstrate its usage by reversing a predefined string using the stack."
                        ],
                        [
                            "Write a Rust program that calculates and prints the first 20 happy numbers. Include a function to check if a number is happy and use it in your solution."
                        ],
                        [
                            "Write a Java program that implements a simple binary search algorithm. Create a sorted array of integers and demonstrate searching for different values, including cases where the value is found and not found."
                        ],
                        [
                            "Write a Go program that generates and prints a multiplication table from 1 to 12. Format the output in a neat grid with proper alignment."
                        ],
                    ],
                    example_labels=[
                        "🏮 春节主题图案",
                        "🎎 日本の伝統的な和柄パターン",
                        "🌐 Particles in a Spherical Container",
                        "💹 NVIDIA Stock Analysis with Matplotlib",
                        "🧮 Calculator with Gradio",
                        "📝 Todo List App with React.js",
                        "📂 File Upload Web App with Streamlit",
                        "💦 Solve Trapping Rain Water Problem",
                        "🎮 Pygame Bouncing Ball Game",
                        "💳 Financial Dashboard with Vue.js",
                        "🔑 User Login Process Flowchart",
                        "🔢 Fibonacci Sequence with Tests",
                        "📌 Vue Kanban Board",
                        "🧹 Streamlit Data Cleaning App",
                        "⚗️ Interactive Periodic Table with React",
                        "📚 Dictionary Sorting Tests in Python",
                        "🏋️‍♂️ Fitness Tracker with React",
                        "🖥️ Vue Server Monitoring",
                        "🔢 Prime Numbers in C",
                        "🧮 OOP Calculator in C++",
                        "🔷 Pascal's Triangle in Rust",
                        "🏛️ Bank Account Simulation in Java",
                        "🐰 Fibonacci Sequence in Go",
                        "📊 Letter Frequency Histogram in C",
                        "📦 Stack Implementation in C++",
                        "😄 Happy Numbers in Rust",
                        "🔎 Binary Search in Java",
                        "✖️ Multiplication Table in Go",
                    ],
                    examples_per_page=100,
                    label="Example Prompts",
                    inputs=[text_input],
                )
            # Ranking Tab
            with gr.Tab("📊 Ranking", id="ranking"):
                gr.Markdown("## 🏆 Model Leaderboard")
                gr.Markdown("*Rankings auto-refresh every 10 minutes*")

                ranking_table = gr.Dataframe(
                    headers=[
                        "Rank",
                        "Model",
                        "Win Rate (%)",
                        "Wins",
                        "Losses",
                        "Ties",
                        "Total Battles",
                    ],
                    datatype=[
                        "number",
                        "str",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                    ],
                    label="Model Rankings",
                    interactive=False,
                    wrap=True,
                )

                ranking_last_update = gr.Markdown("**Last Updated:** Not loaded yet")

                # Timer for auto-refresh every 10 minutes
                ranking_timer = gr.Timer(value=600.0, active=True)

        # Event handlers
        # Create state variables for the run buttons
        state0_var = gr.State()
        state1_var = gr.State()

        # Add telemetry logging for user interactions after state variables are created
        # We need to create a wrapper function to extract the sandbox state from the main state
        def log_telemetry_a(state0, sandbox_ui):
            if state0 and "sandbox_state" in state0:
                # Print user interactions for debugging
                if sandbox_ui and len(sandbox_ui) > 2:
                    interactions = sandbox_ui[2]  # Third element is user_interaction_records
                    if interactions:
                        # Store interactions in the state
                        if "interactions" not in state0:
                            state0["interactions"] = []
                        state0["interactions"].extend(interactions)
                return log_sandbox_telemetry_gradio_fn(state0["sandbox_state"], sandbox_ui)
            return None
            
        def log_telemetry_b(state1, sandbox_ui):
            if state1 and "sandbox_state" in state1:
                # Print user interactions for debugging
                if sandbox_ui and len(sandbox_ui) > 2:
                    interactions = sandbox_ui[2]  # Third element is user_interaction_records
                    if interactions:
                        # Store interactions in the state
                        if "interactions" not in state1:
                            state1["interactions"] = []
                        state1["interactions"].extend(interactions)
                return log_sandbox_telemetry_gradio_fn(state1["sandbox_state"], sandbox_ui)
            return None
        
        sandbox_component_a.change(
            fn=log_telemetry_a,
            inputs=[state0_var, sandbox_component_a],
        )
        sandbox_component_b.change(
            fn=log_telemetry_b,
            inputs=[state1_var, sandbox_component_b],
        )

        # Create response components (hidden but needed for outputs)
        response_a = gr.Markdown("", visible=False)
        response_b = gr.Markdown("", visible=False)

        # Create a wrapper function that handles both the main execution and state update
        def send_and_update_state(state0, state1, text, temp, max_tok, model_a, model_b):
            
            # Hide vote buttons immediately when generation starts
            initial_vote_visibility = False
            
            # Call the main function
            result = add_text_and_generate(state0, state1, text, temp, max_tok, model_a, model_b)
            # Extract the state from the result
            new_state0, new_state1 = result[0], result[1]

            # Check if both models have output and are not generating to show vote buttons
            show_vote_buttons = (
                new_state0
                and new_state0.get("has_output", False)
                and not new_state0.get("generating", False)
                and new_state1
                and new_state1.get("has_output", False)
                and not new_state1.get("generating", False)
            )

            # Return all the original outputs plus the updated state for run buttons
            # Make sure all outputs are properly formatted for their expected types
            return (
                new_state0,  # state0
                new_state1,  # state1
                result[2],  # chatbot_a (chat0)
                result[3],  # chatbot_b (chat1)
                result[4]["content"] if isinstance(result[4], dict) else result[4],  # response_a (response0)
                result[5]["content"] if isinstance(result[5], dict) else result[5],  # response_b (response1)
                result[6],  # code_a (code0)
                result[7],  # code_b (code1)
                result[10] if len(result) > 10 else "",  # sandbox_state0
                result[11] if len(result) > 11 else "",  # sandbox_state1
                result[12] if len(result) > 12 else "",  # sandbox_output0
                result[13] if len(result) > 13 else "",  # sandbox_output1
                (
                    result[14] if len(result) > 14 else gr.update(visible=False)
                ),  # sandbox_component_update0
                (
                    result[15] if len(result) > 15 else gr.update(visible=False)
                ),  # sandbox_component_update1
                (
                    result[16] if len(result) > 16 else "**Conversation:** 0 turns"
                ),  # chat_stats_a
                (
                    result[17] if len(result) > 17 else "**Conversation:** 0 turns"
                ),  # chat_stats_b
                result[18] if len(result) > 18 else "",  # sandbox_view_a
                result[19] if len(result) > 19 else "",  # sandbox_view_b
                new_state0,  # state0_var
                new_state1,  # state1_var
                text,  # Keep original text input
                f"**Model A:** {model_a}",  # Update model display A
                f"**Model B:** {model_b}",  # Update model display B
                gr.update(visible=show_vote_buttons),  # vote_section
                gr.update(visible=show_vote_buttons),  # vote_buttons_row
                gr.update(visible=False),  # vote_status
                gr.update(interactive=show_vote_buttons),  # vote_left_btn
                gr.update(interactive=show_vote_buttons),  # vote_right_btn
                gr.update(interactive=show_vote_buttons),  # vote_tie_btn
                gr.update(interactive=show_vote_buttons),  # vote_both_bad_btn
            )

        send_btn.click(
            fn=send_and_update_state,
            inputs=[
                state0_var,  # state0
                state1_var,  # state1
                text_input,
                temperature,
                max_tokens,
                gr.State(model_a),  # Use fixed model A
                gr.State(model_b),  # Use fixed model B
            ],
            outputs=[
                state0_var,  # state0
                state1_var,  # state1
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(),  # sandbox_state0
                gr.State(),  # sandbox_state1
                sandbox_view_a,  # sandbox output for model A
                sandbox_view_b,  # sandbox output for model B
                sandbox_component_a,  # sandbox component for model A
                sandbox_component_b,  # sandbox component for model B
                chat_stats_a,  # Conversation statistics for model A
                chat_stats_b,  # Conversation statistics for model B
                sandbox_view_a,  # Sandbox view for model A
                sandbox_view_b,  # Sandbox view for model B
                state0_var,  # Updated state for run button A
                state1_var,  # Updated state for run button B
                text_input,  # Clear the text input after sending
                model_display_a,  # Update model display A
                model_display_b,  # Update model display B
                vote_section,  # Show/hide vote section
                vote_buttons_row,  # Show/hide vote buttons
                vote_status,  # Vote status message
                vote_left_btn,  # vote_left_btn
                vote_right_btn,  # vote_right_btn
                vote_tie_btn,  # vote_tie_btn
                vote_both_bad_btn,  # vote_both_bad_btn
            ],
        )

        # Add Enter key submission support to textbox
        text_input.submit(
            fn=send_and_update_state,
            inputs=[
                state0_var,  # state0
                state1_var,  # state1
                text_input,
                temperature,
                max_tokens,
                gr.State(model_a),  # Use fixed model A
                gr.State(model_b),  # Use fixed model B
            ],
            outputs=[
                state0_var,  # state0
                state1_var,  # state1
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(),  # sandbox_state0
                gr.State(),  # sandbox_state1
                sandbox_view_a,  # sandbox output for model A
                sandbox_view_b,  # sandbox output for model B
                sandbox_component_a,  # sandbox component for model A
                sandbox_component_b,  # sandbox component for model B
                chat_stats_a,  # Conversation statistics for model A
                chat_stats_b,  # Conversation statistics for model B
                sandbox_view_a,  # Sandbox view for model A
                sandbox_view_b,  # Sandbox view for model B
                state0_var,  # Updated state for run button A
                state1_var,  # Updated state for run button B
                text_input,  # Clear the text input after sending
                model_display_a,  # Update model display A
                model_display_b,  # Update model display B
                vote_section,  # Show/hide vote section
                vote_buttons_row,  # Show/hide vote buttons
                vote_status,  # Vote status message
                vote_left_btn,  # vote_left_btn
                vote_right_btn,  # vote_right_btn
                vote_tie_btn,  # vote_tie_btn
                vote_both_bad_btn,  # vote_both_bad_btn
            ],
        )

        # Retry button handler
        retry_btn.click(
            fn=retry_last_message,
            inputs=[
                state0_var,  # state0
                state1_var,  # state1
                gr.State(model_a),  # Use fixed model A
                gr.State(model_b),  # Use fixed model B
            ],
            outputs=[
                state0_var,  # state0
                state1_var,  # state1
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(),  # sandbox_state0
                gr.State(),  # sandbox_state1
                sandbox_view_a,  # sandbox output for model A
                sandbox_view_b,  # sandbox output for model B
                sandbox_component_a,  # sandbox component for model A
                sandbox_component_b,  # sandbox component for model B
                chat_stats_a,  # Conversation statistics for model A
                chat_stats_b,  # Conversation statistics for model B
                sandbox_view_a,  # Sandbox view for model A
                sandbox_view_b,  # Sandbox view for model B
                state0_var,  # Updated state for run button A
                state1_var,  # Updated state for run button B
                text_input,  # Clear the text input after sending
                model_display_a,  # Update model display A
                model_display_b,  # Update model display B
                vote_section,  # Show/hide vote section
                vote_buttons_row,  # Show/hide vote buttons
                vote_status,  # Vote status message
                vote_left_btn,  # vote_left_btn
                vote_right_btn,  # vote_right_btn
                vote_tie_btn,  # vote_tie_btn
                vote_both_bad_btn,  # vote_both_bad_btn
            ],
        )

        # Send left button handler
        send_left_btn.click(
            fn=send_to_left_only,
            inputs=[
                state0_var,  # state0
                state1_var,  # state1
                text_input,
                temperature,
                max_tokens,
                gr.State(model_a),  # Use fixed model A
                gr.State(model_b),  # Use fixed model B
            ],
            outputs=[
                state0_var,  # state0
                state1_var,  # state1
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(),  # sandbox_state0
                gr.State(),  # sandbox_state1
                sandbox_view_a,  # sandbox output for model A
                sandbox_view_b,  # sandbox output for model B
                sandbox_component_a,  # sandbox component for model A
                sandbox_component_b,  # sandbox component for model B
                chat_stats_a,  # Conversation statistics for model A
                chat_stats_b,  # Conversation statistics for model B
                sandbox_view_a,  # Sandbox view for model A
                sandbox_view_b,  # Sandbox view for model B
                state0_var,  # Updated state for run button A
                state1_var,  # Updated state for run button B
                text_input,  # Clear the text input after sending
                model_display_a,  # Update model display A
                model_display_b,  # Update model display B
                vote_section,  # Show/hide vote section
                vote_buttons_row,  # Show/hide vote buttons
                vote_status,  # Vote status message
                vote_left_btn,  # vote_left_btn
                vote_right_btn,  # vote_right_btn
                vote_tie_btn,  # vote_tie_btn
                vote_both_bad_btn,  # vote_both_bad_btn
            ],
        )

        # Send right button handler
        send_right_btn.click(
            fn=send_to_right_only,
            inputs=[
                state0_var,  # state0
                state1_var,  # state1
                text_input,
                temperature,
                max_tokens,
                gr.State(model_a),  # Use fixed model A
                gr.State(model_b),  # Use fixed model B
            ],
            outputs=[
                state0_var,  # state0
                state1_var,  # state1
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(),  # sandbox_state0
                gr.State(),  # sandbox_state1
                sandbox_view_a,  # sandbox output for model A
                sandbox_view_b,  # sandbox output for model B
                sandbox_component_a,  # sandbox component for model A
                sandbox_component_b,  # sandbox component for model B
                chat_stats_a,  # Conversation statistics for model A
                chat_stats_b,  # Conversation statistics for model B
                sandbox_view_a,  # Sandbox view for model A
                sandbox_view_b,  # Sandbox view for model B
                state0_var,  # Updated state for run button A
                state1_var,  # Updated state for run button B
                text_input,  # Clear the text input after sending
                model_display_a,  # Update model display A
                model_display_b,  # Update model display B
                vote_section,  # Show/hide vote section
                vote_buttons_row,  # Show/hide vote buttons
                vote_status,  # Vote status message
                vote_left_btn,  # vote_left_btn
                vote_right_btn,  # vote_right_btn
                vote_tie_btn,  # vote_tie_btn
                vote_both_bad_btn,  # vote_both_bad_btn
            ],
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[state0_var, state1_var],
            outputs=[
                state0_var,      # Reset state0
                state1_var,      # Reset state1
                chatbot_a,       # Clear chatbot_a
                chatbot_b,       # Clear chatbot_b
                response_a,      # Clear response_a
                response_b,      # Clear response_b
                code_a,          # Clear code_a
                code_b,          # Clear code_b
                gr.State(None),  # Reset sandbox_state0
                gr.State(None),  # Reset sandbox_state1
                sandbox_view_a,  # Clear sandbox_view_a
                sandbox_view_b,  # Clear sandbox_view_b
                sandbox_component_a,  # Hide sandbox_component_a
                sandbox_component_b,  # Hide sandbox_component_b
                chat_stats_a,    # Reset conversation statistics for model A
                chat_stats_b,    # Reset conversation statistics for model B
                sandbox_view_a,  # Reset sandbox view for model A
                sandbox_view_b,  # Reset sandbox view for model B
                model_display_a, # Reset model display A
                model_display_b, # Reset model display B
                text_input,      # Clear text input
                vote_section,    # Hide vote section
                vote_buttons_row, # Hide vote buttons
                vote_status,     # Clear vote status
                vote_left_btn,   # Disable vote buttons
                vote_right_btn,  # Disable vote buttons
                vote_tie_btn,    # Disable vote buttons
                vote_both_bad_btn, # Disable vote buttons
            ]
        )

        # Refresh models button handler
        def refresh_models():
            new_model_a, new_model_b = get_random_models()
            return (
                None,  # Reset state0
                None,  # Reset state1
                "",  # Clear chat A
                "",  # Clear chat B
                "",  # Clear response A
                "",  # Clear response B
                "",  # Clear code A
                "",  # Clear code B
                gr.State(None),  # Reset sandbox state A
                gr.State(None),  # Reset sandbox state B
                "",  # Clear sandbox view A
                "",  # Clear sandbox view B
                gr.update(visible=False),  # Hide sandbox component A
                gr.update(visible=False),  # Hide sandbox component B
                "**Conversation:** 0 turns | **Total Messages:** 0",  # Reset stats A
                "**Conversation:** 0 turns | **Total Messages:** 0",  # Reset stats B
                "",  # Clear sandbox view A
                "",  # Clear sandbox view B
                None,  # Reset state0_var
                None,  # Reset state1_var
                f"**Model A:** {new_model_a}",  # Update model display A
                f"**Model B:** {new_model_b}",  # Update model display B
                gr.update(visible=False),  # Hide vote section
                gr.update(visible=False),  # Hide vote buttons
                gr.update(visible=False),  # Clear vote status
            )

        refresh_models_btn.click(
            fn=refresh_models,
            inputs=[],
            outputs=[
                state0_var,
                state1_var,
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(None),
                gr.State(None),
                sandbox_view_a,
                sandbox_view_b,
                sandbox_component_a,
                sandbox_component_b,
                chat_stats_a,
                chat_stats_b,
                sandbox_view_a,
                sandbox_view_b,
                state0_var,
                state1_var,
                model_display_a,  # Update model display A
                model_display_b,  # Update model display B
                vote_section,  # Hide vote section
                vote_buttons_row,  # Hide vote buttons
                vote_status,  # Clear vote status
            ],
        )

        # Vote button handlers
        def vote_and_clear(state0, state1, vote_type):
            # First save the vote (now runs in background)
            message, ranking_update, last_update = handle_vote(
                state0, state1, vote_type
            )
            
            # Get the model names from the current session
            model_a = state0["model_name"] if state0 else "Unknown"
            model_b = state1["model_name"] if state1 else "Unknown"
            
            # Always show thank you message and clear everything immediately
            gr.Info("Thank you for your vote! 🎉 Your feedback has been recorded and new models have been selected.", duration=5)

            # revval the model names in the info message
            gr.Info(f"Now you can see model names! 👀 \nModel A: {model_a}, Model B: {model_b}", duration=15)
            
            # Get new random models for the next session
            model_a, model_b = get_random_models()
            
            # Clear everything and start fresh immediately, but preserve examples
            return (
                "Thank you for your vote! 🎉",  # vote status with thank you message
                None,  # Clear state0
                None,  # Clear state1
                "",  # Clear chatbot_a
                "",  # Clear chatbot_b
                "",  # Clear response_a
                "",  # Clear response_b
                "",  # Clear code_a
                "",  # Clear code_b
                "",  # Clear sandbox_view_a
                "",  # Clear sandbox_view_b
                gr.update(visible=False),  # Hide sandbox_component_a
                gr.update(visible=False),  # Hide sandbox_component_b
                "**Conversation:** 0 turns | **Total Messages:** 0",  # Reset chat_stats_a
                "**Conversation:** 0 turns | **Total Messages:** 0",  # Reset chat_stats_b
                f"**Model A:** {model_a}",  # Update model_display_a
                f"**Model B:** {model_b}",  # Update model_display_b
                gr.update(visible=False),  # Hide vote_section
                gr.update(visible=False),  # Hide vote_buttons_row
                None,  # Reset state0_var
                None,  # Reset state1_var
                gr.update(),  # Keep existing ranking_table (no refresh needed)
                gr.update(),  # Keep existing ranking_last_update (no refresh needed)
                gr.update(interactive=False),  # Disable vote_left_btn
                gr.update(interactive=False),  # Disable vote_right_btn
                gr.update(interactive=False),  # Disable vote_tie_btn
                gr.update(interactive=False),  # Disable vote_both_bad_btn
                "",  # Clear text_input to preserve examples
            )

        # Vote button click handlers
        for vote_btn, vote_type in [
            (vote_left_btn, "left"),
            (vote_right_btn, "right"),
            (vote_tie_btn, "tie"),
            (vote_both_bad_btn, "both_bad"),
        ]:
            vote_btn.click(
                fn=vote_and_clear,
                inputs=[state0_var, state1_var, gr.State(vote_type)],
                outputs=[
                    vote_status,  # vote status message
                    state0_var,  # state0
                    state1_var,  # state1
                    chatbot_a,  # chatbot_a
                    chatbot_b,  # chatbot_b
                    response_a,  # response_a
                    response_b,  # response_b
                    code_a,  # code_a
                    code_b,  # code_b
                    sandbox_view_a,  # sandbox_view_a
                    sandbox_view_b,  # sandbox_view_b
                    sandbox_component_a,  # sandbox_component_a
                    sandbox_component_b,  # sandbox_component_b
                    chat_stats_a,  # chat_stats_a
                    chat_stats_b,  # chat_stats_b
                    model_display_a,  # model_display_a
                    model_display_b,  # model_display_b
                    vote_section,  # vote_section
                    vote_buttons_row,  # vote_buttons_row
                    state0_var,  # state0_var (duplicate for state management)
                    state1_var,  # state1_var (duplicate for state management)
                    ranking_table,  # ranking_table
                    ranking_last_update,  # ranking_last_update
                    vote_left_btn,  # vote_left_btn
                    vote_right_btn,  # vote_right_btn
                    vote_tie_btn,  # vote_tie_btn
                    vote_both_bad_btn,  # vote_both_bad_btn
                    text_input,  # text_input (to preserve examples)
                ],
            )

        # Ranking handlers
        def update_ranking_display():
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

    return demo

def main():
    """Main function to run the Simple BigCodeArena app"""
    # Get random models for this session
    model_a, model_b = get_random_models()

    # Build the UI
    demo = build_ui()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()
