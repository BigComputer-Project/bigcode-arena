"""
Simple BigCodeArena - A simplified AI coding battle arena
Focuses on core functionality: two models, automatic code extraction, and execution
"""

import gradio as gr
from gradio_sandboxcomponent import SandboxComponent

# Import completion utilities
from completion import make_config, registered_api_completion

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
        print(f"Error loading API config: {e}")
        return {}

# Global variables
api_config = load_api_config()
available_models = list(api_config.keys()) if api_config else []

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
        "sandbox_state": create_sandbox_state()
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
    messages = []
    for msg in state["messages"]:
        if msg["role"] in ["user", "assistant"] and msg["content"] is not None:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Get model config
    model_name = state["model_name"]
    if model_name not in api_config:
        print(f"Model {model_name} not found in config")
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
        return state, response_text
    else:
        error_msg = f"Error: Invalid response format from {api_type}"
        print(error_msg)
        return state, error_msg

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
    
    code, code_language, code_dependencies, env_selection = extract_result
    
    # Update sandbox state (now a dictionary)
    sandbox_state['code_to_execute'] = code
    sandbox_state['code_dependencies'] = code_dependencies
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
        print(f"Models: {state0['model_name']} vs {state1['model_name']}")
    
    # Add user message to both states
    state0["messages"].append({"role": "user", "content": text})
    state1["messages"].append({"role": "user", "content": text})
    
    # Generate responses
    state0, response0 = generate_response_with_completion(state0, temperature, max_tokens)
    state1, response1 = generate_response_with_completion(state1, temperature, max_tokens)
    
    # Add the assistant responses to the message history
    state0["messages"].append({"role": "assistant", "content": response0})
    state1["messages"].append({"role": "assistant", "content": response1})
    
    # Format chat history for display
    chat0 = format_chat_history(state0["messages"])
    chat1 = format_chat_history(state1["messages"])
    
    # Extract code from responses for sandbox
    sandbox_state0 = state0.get("sandbox_state", create_sandbox_state())
    sandbox_state1 = state1.get("sandbox_state", create_sandbox_state())
    
    _, code0, env0 = extract_and_execute_code(response0, sandbox_state0)
    _, code1, env1 = extract_and_execute_code(response1, sandbox_state1)
    
    # Update sandbox states in the main states
    state0["sandbox_state"] = sandbox_state0
    state1["sandbox_state"] = sandbox_state1
    
    # Clear previous sandbox outputs when new message is sent
    sandbox_output0 = ""
    sandbox_output1 = ""
    sandbox_component_update0 = gr.update(visible=False)
    sandbox_component_update1 = gr.update(visible=False)
    
    # Also clear the sandbox view components to show fresh results
    sandbox_view_a = ""
    sandbox_view_b = ""
    
    if code0.strip():
        # Get the dependencies from the sandbox state
        dependencies0 = sandbox_state0.get('code_dependencies', ([], []))
        print(f"DEBUG: Running code0 with dependencies: {dependencies0}")
        sandbox_url0, sandbox_output0, sandbox_error0 = run_sandbox_code(sandbox_state0, code0, dependencies0)
        print(f"DEBUG: Code0 result - URL: {sandbox_url0}, Output: {sandbox_output0[:100] if sandbox_output0 else 'None'}, Error: {sandbox_error0[:100] if sandbox_error0 else 'None'}")
        
        # Check if this is a web-based environment that should use SandboxComponent
        env_type = sandbox_state0.get('auto_selected_sandbox_environment') or sandbox_state0.get('sandbox_environment')
        print(f"DEBUG: Model A environment type: {env_type}")
        # Use the URL directly from the function return
        if sandbox_url0:
            sandbox_component_update0 = gr.update(value=(sandbox_url0, True, []), visible=True)
        
        # Update sandbox view with output and errors
        if sandbox_output0:
            sandbox_view_a += f"# Output\n{sandbox_output0}"
        if sandbox_error0:
            sandbox_view_a += f"# Errors\n{sandbox_error0}"
    
    if code1.strip():
        # Get the dependencies from the sandbox state
        dependencies1 = sandbox_state1.get('code_dependencies', ([], []))
        print(f"DEBUG: Running code1 with dependencies: {dependencies1}")
        sandbox_url1, sandbox_output1, sandbox_error1 = run_sandbox_code(sandbox_state1, code1, dependencies1)
        print(f"DEBUG: Code1 result - URL: {sandbox_url1}, Output: {sandbox_output1[:100] if sandbox_output1 else 'None'}, Error: {sandbox_error1[:100] if sandbox_error1 else 'None'}")
        
        # Check if this is a web-based environment that should use SandboxComponent
        env_type = sandbox_state1.get('auto_selected_sandbox_environment') or sandbox_state1.get('sandbox_environment')
        print(f"DEBUG: Model B environment type: {env_type}")
        # Use the URL directly from the function return
        if sandbox_url1:
            sandbox_component_update1 = gr.update(value=(sandbox_url1, True, []), visible=True)
        
        if sandbox_output1:
            sandbox_view_b += f"## Output\n{sandbox_output1}"
        if sandbox_error1:
            sandbox_view_b += f"## Errors\n{sandbox_error1}"
    
    # Calculate conversation statistics
    turn_count_a = len([msg for msg in state0["messages"] if msg["role"] == "assistant" and msg["content"]])
    turn_count_b = len([msg for msg in state1["messages"] if msg["role"] == "assistant" and msg["content"]])
    
    # Format conversation statistics
    chat_stats_a = f"**Conversation:** {turn_count_a} turns | **Total Messages:** {len(state0['messages'])}"
    chat_stats_b = f"**Conversation:** {turn_count_b} turns | **Total Messages:** {len(state1['messages'])}"
    
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
    if state1 and "sandbox_state" in state1:
        reset_sandbox_state(state1["sandbox_state"])
    
    # Get current model names for display
    model_a, model_b = get_random_models()
    
    return None, None, "", "", "", "", "", "", "", "", "", "", "", "", gr.update(visible=False), gr.update(visible=False), "**Conversation:** 0 turns | **Total Messages:** 0", "**Conversation:** 0 turns | **Total Messages:** 0", "", "", f"**Model A:** {model_a}", f"**Model B:** {model_b}"

def run_sandbox_code(sandbox_state: dict, code: str, dependencies: tuple) -> tuple[str, str, str]:
    """Run code in the appropriate sandbox environment"""
    if not code.strip():
        return "", "", "No code to run"
    
    # Update sandbox state
    sandbox_state['code_to_execute'] = code
    sandbox_state['code_dependencies'] = dependencies
    
    # Determine environment
    env = sandbox_state.get('auto_selected_sandbox_environment') or sandbox_state.get('sandbox_environment')
    
    try:
        if env == SandboxEnvironment.HTML:
            sandbox_url, sandbox_id, stderr = run_html_sandbox(code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr
            
        elif env == SandboxEnvironment.REACT:
            result = run_react_sandbox(code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = result['sandbox_id']
            return result['sandbox_url'], "", result['stderr']
            
        elif env == SandboxEnvironment.VUE:
            result = run_vue_sandbox(code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = result['sandbox_id']
            return result['sandbox_url'], "", result['stderr']
            
        elif env == SandboxEnvironment.PYGAME:
            result = run_pygame_sandbox(code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = result['sandbox_id']
            return result['sandbox_url'], "", result['stderr']
            
        elif env == SandboxEnvironment.GRADIO:
            sandbox_url, sandbox_id, stderr = run_gradio_sandbox(code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr
            
        elif env == SandboxEnvironment.STREAMLIT:
            sandbox_url, sandbox_id, stderr = run_streamlit_sandbox(code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr
            
        elif env == SandboxEnvironment.MERMAID:
            # Convert Mermaid to HTML and run in HTML sandbox
            html_code = mermaid_to_html(code, theme='light')
            sandbox_url, sandbox_id, stderr = run_html_sandbox(html_code, dependencies, sandbox_state.get('sandbox_id'))
            sandbox_state['sandbox_id'] = sandbox_id
            return sandbox_url, "", stderr
            
        elif env == SandboxEnvironment.PYTHON_RUNNER:
            output, stderr = run_code_interpreter(code, 'python', dependencies)
            return "", output, stderr
            
        elif env == SandboxEnvironment.JAVASCRIPT_RUNNER:
            html_code = javascript_to_html(code)
            output, stderr = run_html_sandbox(html_code, dependencies, sandbox_state.get('sandbox_id'))
            return "", output, stderr
            
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
            output, stderr = run_code_interpreter(code, 'python', dependencies)
            return "", output, stderr
            
    except Exception as e:
        return "", "", str(e)



def build_ui():
    """Build a UI for the coding arena with integrated sandbox"""
    
    # Get random models for this session
    model_a, model_b = get_random_models()
    
    with gr.Blocks(title="BigCodeArena") as demo:
        gr.Markdown("# BigCodeArena - Start Your Vibe Coding!")
        
        # Model display (non-interactive)
        with gr.Row():
            with gr.Column():
                model_display_a = gr.Markdown(f"**Model A:** {model_a}", visible=False)
            with gr.Column():
                model_display_b = gr.Markdown(f"**Model B:** {model_b}", visible=False)
        
        # Sandbox section with tabs for each model - Collapsible and open by default
        with gr.Accordion("🏗️ Code Execution & Sandbox", open=True):
            
            with gr.Row():
                # Model A Sandbox
                with gr.Column():
                    gr.Markdown("### Model A Sandbox")
                    with gr.Tabs():
                        with gr.Tab("View"):
                            sandbox_view_a = gr.Markdown("**Sandbox output will appear here automatically**")
                            sandbox_component_a = SandboxComponent(
                                value=("", False, []),
                                label="Model A Sandbox",
                                visible=False
                            )
                        with gr.Tab("Code"):
                            code_a = gr.Code(
                                label="Extracted Code",
                                language="python",
                                lines=8,
                                interactive=False
                            )
                
                # Model B Sandbox
                with gr.Column():
                    gr.Markdown("### Model B Sandbox")
                    with gr.Tabs():
                        with gr.Tab("View"):
                            sandbox_view_b = gr.Markdown("**Sandbox output will appear here automatically**")
                            sandbox_component_b = SandboxComponent(
                                value=("", False, []),
                                label="Model B Sandbox",
                                visible=False
                            )
                        with gr.Tab("Code"):
                            code_b = gr.Code(
                                label="Extracted Code",
                                language="python",
                                lines=8,
                                interactive=False
                            )

        # Main chat interface - Collapsible and hidden by default
        with gr.Accordion("💬 Chat Interface", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Model A")
                    chatbot_a = gr.Chatbot(
                        label="Model A",
                        height=300,
                        show_copy_button=True,
                        type="messages"
                    )
                    chat_stats_a = gr.Markdown("**Conversation:** 0 turns")
                
                with gr.Column():
                    gr.Markdown("## Model B")
                    chatbot_b = gr.Chatbot(
                        label="Model B", 
                        height=300,
                        show_copy_button=True,
                        type="messages"
                    )
                    chat_stats_b = gr.Markdown("**Conversation:** 0 turns")
        
        # Input section
        with gr.Row():
            text_input = gr.Textbox(
                label="Enter your coding prompt",
                placeholder="e.g., 'Write a Python function to calculate fibonacci numbers'",
                lines=1
            )
        
        # Control buttons
        with gr.Row():
            send_btn = gr.Button("🚀 Send to Both Models", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            refresh_models_btn = gr.Button("🔄 New Random Models", variant="secondary")
        
        # Advanced Settings (Collapsible)
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                with gr.Column(scale=1):
                    max_tokens = gr.Slider(
                        minimum=500,
                        maximum=32768,
                        value=2048,
                        label="Max Tokens"
                    )
        
        # Event handlers
        # Create state variables for the run buttons
        state0_var = gr.State()
        state1_var = gr.State()
        
        # Create response components (hidden but needed for outputs)
        response_a = gr.Markdown("", visible=False)
        response_b = gr.Markdown("", visible=False)
        
        # Create a wrapper function that handles both the main execution and state update
        def send_and_update_state(state0, state1, text, temp, max_tok, model_a, model_b):
            print(f"DEBUG: send_and_update_state called with text: {text[:50] if text else 'None'}")
            # Call the main function
            result = add_text_and_generate(state0, state1, text, temp, max_tok, model_a, model_b)
            # Extract the state from the result
            new_state0, new_state1 = result[0], result[1]
            print(f"DEBUG: send_and_update_state returning new_state0: {type(new_state0)}, new_state1: {type(new_state1)}")
            # Return all the original outputs plus the updated state for run buttons
            # Make sure all outputs are properly formatted for their expected types
            return (
                new_state0,      # state0
                new_state1,      # state1
                result[2],       # chatbot_a (chat0)
                result[3],       # chatbot_b (chat1)
                result[4],       # response_a (response0)
                result[5],       # response_b (response1)
                result[6],       # code_a (code0)
                result[7],       # code_b (code1)
                result[10],      # sandbox_state0
                result[11],      # sandbox_state1
                result[12],      # sandbox_output0
                result[13],      # sandbox_output1
                result[14],      # sandbox_component_update0
                result[15],      # sandbox_component_update1
                result[16],      # chat_stats_a
                result[17],      # chat_stats_b
                result[18],      # sandbox_view_a
                result[19],      # sandbox_view_b
                new_state0,      # state0_var
                new_state1,      # state1_var
                "",              # Clear text input
                f"**Model A:** {model_a}",  # Update model display A
                f"**Model B:** {model_b}",  # Update model display B
            )
        
        send_btn.click(
            fn=send_and_update_state,
            inputs=[
                state0_var,      # state0
                state1_var,      # state1
                text_input,
                temperature,
                max_tokens,
                gr.State(model_a),  # Use fixed model A
                gr.State(model_b)   # Use fixed model B
            ],
            outputs=[
                state0_var,      # state0
                state1_var,      # state1
                chatbot_a,
                chatbot_b,
                response_a,
                response_b,
                code_a,
                code_b,
                gr.State(),      # sandbox_state0
                gr.State(),      # sandbox_state1
                sandbox_view_a,  # sandbox output for model A
                sandbox_view_b,  # sandbox output for model B
                sandbox_component_a,  # sandbox component for model A
                sandbox_component_b,  # sandbox component for model B
                chat_stats_a,    # Conversation statistics for model A
                chat_stats_b,    # Conversation statistics for model B
                sandbox_view_a,  # Sandbox view for model A
                sandbox_view_b,  # Sandbox view for model B
                state0_var,      # Updated state for run button A
                state1_var,      # Updated state for run button B
                text_input,      # Clear the text input after sending
                model_display_a, # Update model display A
                model_display_b, # Update model display B
            ]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[gr.State(), gr.State()],
            outputs=[
                gr.State(None),
                gr.State(None),
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
                state0_var,      # Reset state for run button A
                state1_var,      # Reset state for run button B
                chat_stats_a,    # Reset conversation statistics for model A
                chat_stats_b,    # Reset conversation statistics for model B
                sandbox_view_a,  # Reset sandbox view for model A
                sandbox_view_b,  # Reset sandbox view for model B
                model_display_a, # Reset model display A
                model_display_b, # Reset model display B
            ]
        )
        
        # Refresh models button handler
        def refresh_models():
            new_model_a, new_model_b = get_random_models()
            return (
                None,  # Reset state0
                None,  # Reset state1
                "",    # Clear chat A
                "",    # Clear chat B
                "",    # Clear response A
                "",    # Clear response B
                "",    # Clear code A
                "",    # Clear code B
                gr.State(None),  # Reset sandbox state A
                gr.State(None),  # Reset sandbox state B
                "",    # Clear sandbox view A
                "",    # Clear sandbox view B
                gr.update(visible=False),  # Hide sandbox component A
                gr.update(visible=False),  # Hide sandbox component B
                "**Conversation:** 0 turns | **Total Messages:** 0",  # Reset stats A
                "**Conversation:** 0 turns | **Total Messages:** 0",  # Reset stats B
                "",    # Clear sandbox view A
                "",    # Clear sandbox view B
                None,  # Reset state0_var
                None,  # Reset state1_var
                f"**Model A:** {new_model_a}",  # Update model display A
                f"**Model B:** {new_model_b}",  # Update model display B
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
            ]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["使用SVG绘制春节主题的动态图案，包括：1）一个红色的灯笼，带有金色的流苏 2）一个金色的福字，使用书法字体 3）背景添加一些烟花效果 4）在灯笼和福字周围添加一些祥云图案。确保图案布局美观，颜色搭配符合春节传统风格。"],
                ["SVGを使用して日本の伝統的な和柄パターンを描画してください。1）波紋（さざなみ）模様 2）市松模様 3）麻の葉模様 4）雷文（らいもん）模様を含めてください。色は伝統的な日本の色（藍色、朱色、金色など）を使用し、レイアウトはバランスよく配置してください。"],
                ["Write HTML with P5.js that simulates 25 particles in a vacuum space of a cylindrical container, bouncing within its boundaries. Use different colors for each ball and ensure they leave a trail showing their movement. Add a slow rotation of the container to give better view of what's going on in the scene. Make sure to create proper collision detection and physic rules to ensure particles remain in the container. Add an external spherical container. Add a slow zoom in and zoom out effect to the whole scene."],
                ["Write a Python script to scrape NVIDIA's stock price for the past month using the yfinance library. Clean the data and create an interactive visualization using Matplotlib. Include: 1) A candlestick chart showing daily price movements 2) A line chart with 7-day and 30-day moving averages. Add hover tooltips showing exact values and date. Make the layout professional with proper titles and axis labels."],
                ["Write a Python script that uses the Gradio library to create a functional calculator. The calculator should support basic arithmetic operations: addition, subtraction, multiplication, and division. It should have two input fields for numbers and a dropdown menu to select the operation."],
                ["Write a Todo list app using React.js. The app should allow users to add, delete, and mark tasks as completed. Include features like filtering tasks by status (completed, active), sorting tasks by priority, and displaying the total number of tasks."],
                ["Write a Python script using the Streamlit library to create a web application for uploading and displaying files. The app should allow users to upload files of type .csv or .txt. If a .csv file is uploaded, display its contents as a table using Streamlit's st.dataframe() method. If a .txt file is uploaded, display its content as plain text."],
                ["Write a Python function to solve the Trapping Rain Water problem. The function should take a list of non-negative integers representing the height of bars in a histogram and return the total amount of water trapped between the bars after raining. Use an efficient algorithm with a time complexity of O(n)."],
                ["Create a simple Pygame script for a game where the player controls a bouncing ball that changes direction when it collides with the edges of the window. Add functionality for the player to control a paddle using arrow keys, aiming to keep the ball from touching the bottom of the screen. Include basic collision detection and a scoring system that increases as the ball bounces off the paddle. You need to add clickable buttons to start the game, and reset the game."],
                ["Create a financial management Dashboard using Vue.js, focusing on local data handling without APIs. Include features like a clean dashboard for tracking income and expenses, dynamic charts for visualizing finances, and a budget planner. Implement functionalities for adding, editing, and deleting transactions, as well as filtering by date or category. Ensure responsive design and smooth user interaction for an intuitive experience."],
                ["Create a Mermaid diagram to visualize a flowchart of a user login process. Include the following steps: User enters login credentials; Credentials are validated; If valid, the user is directed to the dashboard; If invalid, an error message is shown, and the user can retry or reset the password."],
                ["Write a Python function to calculate the Fibonacci sequence up to n numbers. Then write test cases to verify the function works correctly for edge cases like negative numbers, zero, and large inputs."],
                ["Build an HTML page for a Kanban board with three columns with Vue.js: To Do, In Progress, and Done. Each column should allow adding, moving, and deleting tasks. Implement drag-and-drop functionality using Vue Draggable and persist the state using Vuex."],
                ["Develop a Streamlit app that takes a CSV file as input and provides: 1) Basic statistics about the data 2) Interactive visualizations using Plotly 3) A data cleaning interface with options to handle missing values 4) An option to download the cleaned data."],
                ["Write an HTML page with embedded JavaScript that creates an interactive periodic table. Each element should display its properties on hover and allow filtering by category (metals, non-metals, etc.). Include a search bar to find elements by name or symbol."],
                ["Here's a Python function that sorts a list of dictionaries by a specified key:\n\n```python\ndef sort_dicts(data, key):\n    return sorted(data, key=lambda x: x[key])\n```\n\nWrite test cases to verify the function works correctly for edge cases like empty lists, missing keys, and different data types. If you use unittest, please use `unittest.main(argv=['first-arg-is-ignored'], exit=False)` to run the tests."],
                ["Create a React component for a fitness tracker that shows: 1) Daily step count 2) Calories burned 3) Distance walked 4) A progress bar for daily goals."],
                ["Build a Vue.js dashboard for monitoring server health. Include: 1) Real-time CPU and memory usage graphs 2) Disk space visualization 3) Network activity monitor 4) Alerts for critical thresholds."],
                ["Write a C program that calculates and prints the first 100 prime numbers in a formatted table with 10 numbers per row. Include a function to check if a number is prime and use it in your solution."],
                ["Write a C++ program that implements a simple calculator using object-oriented programming. Create a Calculator class with methods for addition, subtraction, multiplication, and division. Include error handling for division by zero."],
                ["Write a Rust program that generates and prints a Pascal's Triangle with 10 rows. Format the output to center-align the numbers in each row."],
                ["Write a Java program that simulates a simple bank account system. Create a BankAccount class with methods for deposit, withdrawal, and balance inquiry. Include error handling for insufficient funds and demonstrate its usage with a few transactions."],
                ["Write a Go program that calculates and prints the Fibonacci sequence up to the 50th number. Format the output in a table with 5 numbers per row and include the index of each Fibonacci number."],
                ["Write a C program that calculates and prints a histogram of letter frequencies from a predefined string. Use ASCII art to display the histogram vertically."],
                ["Write a C++ program that implements a simple stack data structure with push, pop, and peek operations. Demonstrate its usage by reversing a predefined string using the stack."],
                ["Write a Rust program that calculates and prints the first 20 happy numbers. Include a function to check if a number is happy and use it in your solution."],
                ["Write a Java program that implements a simple binary search algorithm. Create a sorted array of integers and demonstrate searching for different values, including cases where the value is found and not found."],
                ["Write a Go program that generates and prints a multiplication table from 1 to 12. Format the output in a neat grid with proper alignment."],
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
        
    return demo

def main():
    """Main function to run the Simple BigCodeArena app"""
    print("🚀 Starting Simple BigCodeArena...")
    if available_models:
        print(f"🔍 Available models: {', '.join(available_models)}")
        # Get random models for this session
        model_a, model_b = get_random_models()
        print(f"🎲 Randomly selected models for this session:")
        print(f"   Model A: {model_a}")
        print(f"   Model B: {model_b}")
    else:
        print("⚠️  No models found in config!")
    
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
