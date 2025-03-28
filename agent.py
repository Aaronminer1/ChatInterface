#!/usr/bin/env python3
"""
Simple Agent Interface using Ollama with local models
"""

import os
import sys
import json
import requests
import subprocess
import tempfile
import uuid
import re
from typing import Dict, List, Any, Generator, Tuple, Optional
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Load environment variables
load_dotenv()

class OllamaAgent:
    """Agent class that interfaces with Ollama API"""
    
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = os.getenv("MODEL_NAME", "gemma3:latest")
        self.available_models = []  # Will be populated by get_available_models
        
        # Known model creators mapping - will be used as a reference for generating identities
        self.known_creators = {
            "gemma": "Google DeepMind",
            "llama": "Meta AI",
            "deepseek": "DeepSeek AI",
            "mistral": "Mistral AI",
            "mixtral": "Mistral AI",
            "phi": "Microsoft",
            "codellama": "Meta AI",
            "dolphin": "Anthropic",
            "neural": "Intel Labs",
            "qwen": "Alibaba Cloud",
            "wizard": "WizardLM team",
            "wizardcoder": "WizardLM team",
            "stable": "Stability AI",
            "falcon": "Technology Innovation Institute",
            "orca": "Microsoft",
            "vicuna": "LMSYS",
            "openchat": "OpenChat team",
            "yi": "01.AI",
            "mpt": "MosaicML",
            "nous": "Nous Research",
            "pythia": "EleutherAI",
            "solar": "Upstage",
            "starcoder": "BigCode",
            "zephyr": "Hugging Face",
        }
        
        # Dynamic model identities - will be populated based on available models
        self.model_identities = {}
    
    def chat(self, message: str, history: List[Dict[str, str]]) -> str:
        """Send a chat message to the Ollama API and get a response"""
        
        # Create a copy of the history
        messages = history.copy()
        
        # Add a system message to inform the model of its identity
        # Check if there's already a system message at the beginning
        if not messages or messages[0].get("role") != "system":
            # Add a system message with the model's identity
            identity_message = self._get_identity_message()
            messages.insert(0, {"role": "system", "content": identity_message})
        elif messages[0].get("role") == "system":
            # Update the existing system message with the current model's identity
            messages[0]["content"] = self._get_identity_message()
        
        # Add the current message to history
        messages.append({"role": "user", "content": message})
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        try:
            # Make the API call to Ollama
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload)
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                return f"Error: Received status code {response.status_code} from Ollama API: {response.text}"
        
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def generate_model_identities(self):
        """Generate identities for all available models based on their names"""
        # Clear existing identities
        self.model_identities = {}
        
        # Generate identities for each available model
        for model_name in self.available_models:
            # Extract the base name (remove version and size info)
            base_name = model_name.split(':')[0].split('-')[0].lower()
            
            # Find creator based on known mappings
            creator = None
            for key, value in self.known_creators.items():
                if key in base_name:
                    creator = value
                    break
            
            # If no specific creator found, use a generic one
            if not creator:
                creator = "an AI research lab"
            
            # Format the model name for display
            display_name = base_name.capitalize()
            
            # Generate the identity string
            identity = f"{display_name}, a large language model created by {creator}"
            
            # Store in the identities dictionary
            self.model_identities[model_name] = identity
            
            print(f"Generated identity for {model_name}: {identity}")
    
    def _get_identity_message(self) -> str:
        """Generate a system message that informs the model of its identity"""
        model_name = self.model_name
        
        # Ensure we have identities generated
        if not self.model_identities:
            self.generate_model_identities()
        
        # Get the model's identity from the mapping, or use a generic identity
        identity = self.model_identities.get(
            model_name, 
            f"{model_name}, a large language model"
        )
        
        # Extract a simple name for the model
        simple_name = model_name.split(':')[0].split('-')[0].capitalize()
        
        # Create a system message that informs the model of its identity
        return f"""You are {identity}. When asked about your identity or who created you, 
        always respond accurately based on this information. Never claim to be a different model.
        Your name is {simple_name}.
        
        You can write and execute code. When you write code, it will be executed and the results 
        will be shown to the user. You can write code in Python, JavaScript, or Bash/Shell.
        
        When writing code, always use the following format:
        ```language
        # Your code here
        ```
        
        For example:
        ```python
        print("Hello, world!")
        ```
        
        The code will be automatically executed, and the results will be shown to the user.
        
        When writing Python code that requires external libraries, you can specify dependencies in two ways:
        1. Add a comment like '# pip install package_name' at the top of your code
        2. Simply import the libraries you need, and the system will automatically detect and install them
        
        For example:
        ```python
        # pip install pandas matplotlib
        import pandas as pd
        import matplotlib.pyplot as plt
        # Your code here
        ```
        
        The system supports many common libraries including numpy, pandas, matplotlib, pygame, requests, flask, and more.
        
        Respond to the user in a helpful, accurate, and thoughtful manner."""
        
    
    def chat_stream(self, message: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Send a chat message to the Ollama API and stream the response"""
        
        # Create a copy of the history
        messages = history.copy()
        
        # Add a system message to inform the model of its identity
        # Check if there's already a system message at the beginning
        if not messages or messages[0].get("role") != "system":
            # Add a system message with the model's identity
            identity_message = self._get_identity_message()
            messages.insert(0, {"role": "system", "content": identity_message})
        elif messages[0].get("role") == "system":
            # Update the existing system message with the current model's identity
            messages[0]["content"] = self._get_identity_message()
        
        # Add the current message to history
        messages.append({"role": "user", "content": message})
        
        # Prepare the request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }
        
        try:
            # Make the API call to Ollama with streaming
            response = requests.post(
                f"{self.base_url}/api/chat",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                stream=True
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                yield chunk['message']['content']
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: Received status code {response.status_code} from Ollama API"
        
        except Exception as e:
            yield f"Error connecting to Ollama: {str(e)}"


# Initialize Flask app
app = Flask(__name__)

# Initialize the agent
agent = OllamaAgent()

# Get available models from Ollama
def get_available_models():
    try:
        response = requests.get(f"{agent.base_url}/api/tags")
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            # Extract model names and sort them
            model_names = sorted([model.get('name') for model in models_data])
            agent.available_models = model_names  # Update the agent's available models
            
            # Generate identities for all available models
            agent.generate_model_identities()
            
            return model_names
        else:
            return [agent.model_name]  # Return default model if can't fetch
    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return [agent.model_name]  # Return default model if error

# Create a template for the chat interface
@app.route('/')
def index():
    available_models = get_available_models()
    return render_template('index.html', model_name=agent.model_name, available_models=available_models)

# API endpoint for chat
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    history = data.get('history', [])
    stream_mode = data.get('stream', True)  # Default to streaming
    model_name = data.get('model', agent.model_name)  # Get model name from request or use default
    
    # Validate the model name
    if not agent.available_models:
        get_available_models()  # Populate available models if empty
    
    if model_name not in agent.available_models and agent.available_models:
        print(f"Warning: Requested model '{model_name}' not found in available models. Using default.")
        model_name = agent.model_name
    
    # Temporarily set the model name for this request
    original_model = agent.model_name
    agent.model_name = model_name
    print(f"Using model: {agent.model_name} for this request")
    
    if not message:
        agent.model_name = original_model  # Restore original model
        return jsonify({'error': 'No message provided'}), 400
    
    if not stream_mode:
        # Non-streaming mode
        response = agent.chat(message, history)
        
        # Restore original model
        agent.model_name = original_model
        
        return jsonify({
            'response': response,
            'model': model_name,
            'history': history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
        })
    else:
        # Streaming mode
        def generate():
            full_response = ""
            for chunk in agent.chat_stream(message, history):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk, 'full': full_response})}\n\n"
            
            # Send the final message with updated history
            final_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": full_response}
            ]
            
            # Restore original model
            agent.model_name = original_model
            
            yield f"data: {json.dumps({'done': True, 'model': model_name, 'history': final_history})}\n\n"
        
        return Response(stream_with_context(generate()), mimetype='text/event-stream')


# API endpoint for streaming chat
@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    data = request.json
    message = data.get('message', '')
    history = data.get('history', [])
    model_name = data.get('model', agent.model_name)  # Get model name from request or use default
    
    # Temporarily set the model name for this request
    original_model = agent.model_name
    agent.model_name = model_name
    
    if not message:
        agent.model_name = original_model  # Restore original model
        return jsonify({'error': 'No message provided'}), 400
    
    def generate():
        full_response = ""
        for chunk in agent.chat_stream(message, history):
            full_response += chunk
            yield f"data: {json.dumps({'chunk': chunk, 'full': full_response})}\n\n"
        
        # Send the final message with updated history
        final_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": full_response}
        ]
        
        # Restore original model
        agent.model_name = original_model
        
        yield f"data: {json.dumps({'done': True, 'model': model_name, 'history': final_history})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# API endpoint to get available models
@app.route('/api/models', methods=['GET'])
def get_models():
    available_models = get_available_models()
    return jsonify({
        'models': available_models, 
        'current': agent.model_name,
        'status': 'success'
    })

# API endpoint for code execution
@app.route('/api/execute', methods=['POST'])
def execute_code():
    data = request.json
    code = data.get('code', '')
    language = data.get('language', 'python')
    
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    # Execute the code
    result, error = execute_code_safely(code, language)
    
    return jsonify({
        'result': result,
        'error': error,
        'language': language
    })


def execute_code_safely(code: str, language: str) -> Tuple[str, Optional[str]]:
    """Execute code in a safe environment and return the result"""
    # Create a unique ID for this execution
    execution_id = str(uuid.uuid4())[:8]
    
    # Default result and error
    result = ""
    error = None
    
    # Create a temporary directory for code execution
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Handle different languages
            if language.lower() == 'python':
                # Check for special dependency comments
                dependency_pattern = re.compile(r'^\s*#\s*pip\s+install\s+(.+)$', re.MULTILINE)
                dependency_matches = dependency_pattern.findall(code)
                explicit_dependencies = []
                for match in dependency_matches:
                    packages = match.strip().split()
                    explicit_dependencies.extend(packages)
                
                # Check for import statements to detect dependencies
                import_pattern = re.compile(r'^\s*import\s+([\w\.,\s]+)|^\s*from\s+([\w\.]+)\s+import', re.MULTILINE)
                matches = import_pattern.findall(code)
                
                # Extract package names from import statements
                packages = []
                for match in matches:
                    # Handle 'import x, y, z' format
                    if match[0]:
                        for pkg in match[0].split(','):
                            # Get the base package name (before any dots)
                            base_pkg = pkg.strip().split('.')[0]
                            if base_pkg and base_pkg not in ['os', 'sys', 're', 'json', 'time', 'math', 'random', 
                                                           'datetime', 'collections', 'itertools', 'functools',
                                                           'typing', 'pathlib', 'uuid', 'tempfile', 'subprocess']:
                                packages.append(base_pkg)
                    
                    # Handle 'from x import y' format
                    if match[1]:
                        # Get the base package name (before any dots)
                        base_pkg = match[1].strip().split('.')[0]
                        if base_pkg and base_pkg not in ['os', 'sys', 're', 'json', 'time', 'math', 'random',
                                                       'datetime', 'collections', 'itertools', 'functools',
                                                       'typing', 'pathlib', 'uuid', 'tempfile', 'subprocess']:
                            packages.append(base_pkg)
                
                # Add common packages for specific imports
                package_mapping = {
                    'pygame': ['pygame'],
                    'np': ['numpy'],
                    'numpy': ['numpy'],
                    'pd': ['pandas'],
                    'pandas': ['pandas'],
                    'plt': ['matplotlib'],
                    'matplotlib': ['matplotlib'],
                    'sk': ['scikit-learn'],
                    'sklearn': ['scikit-learn'],
                    'tf': ['tensorflow'],
                    'torch': ['torch'],
                    'cv2': ['opencv-python'],
                    'bs4': ['beautifulsoup4'],
                    'requests': ['requests'],
                    'flask': ['flask'],
                    'django': ['django'],
                    'sqlalchemy': ['sqlalchemy'],
                    'selenium': ['selenium'],
                    'scrapy': ['scrapy'],
                    'nltk': ['nltk'],
                    'spacy': ['spacy'],
                    'gensim': ['gensim'],
                    'keras': ['keras'],
                    'theano': ['theano'],
                    'bokeh': ['bokeh'],
                    'seaborn': ['seaborn'],
                    'plotly': ['plotly'],
                    'dash': ['dash'],
                    'sympy': ['sympy'],
                    'networkx': ['networkx'],
                    'pillow': ['pillow'],
                    'PIL': ['pillow'],
                }
                
                for pkg in packages:
                    if pkg in package_mapping:
                        packages.extend(package_mapping[pkg])
                
                # Add explicit dependencies from comments
                packages.extend(explicit_dependencies)
                
                # Create a virtual environment for the execution
                venv_path = os.path.join(temp_dir, 'venv')
                subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True, capture_output=True)
                
                # Determine the pip executable path based on the OS
                pip_path = os.path.join(venv_path, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_path, 'Scripts', 'pip.exe')
                python_path = os.path.join(venv_path, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_path, 'Scripts', 'python.exe')
                
                # Install detected packages if any
                if packages:
                    unique_packages = list(set(packages))  # Remove duplicates
                    install_output = "Installing dependencies: " + ", ".join(unique_packages) + "\n"
                    
                    # First upgrade pip
                    try:
                        subprocess.run(
                            [pip_path, 'install', '--upgrade', 'pip'],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=False
                        )
                    except Exception as e:
                        install_output += f"Warning: Error upgrading pip: {str(e)}\n"
                    
                    # Install packages one by one
                    for pkg in unique_packages:
                        try:
                            install_process = subprocess.run(
                                [pip_path, 'install', pkg],
                                capture_output=True,
                                text=True,
                                timeout=60,  # 60 second timeout for installation
                                check=False
                            )
                            if install_process.returncode != 0:
                                install_output += f"Warning: Failed to install {pkg}: {install_process.stderr}\n"
                                # Try with --no-cache-dir option
                                retry_process = subprocess.run(
                                    [pip_path, 'install', '--no-cache-dir', pkg],
                                    capture_output=True,
                                    text=True,
                                    timeout=60,
                                    check=False
                                )
                                if retry_process.returncode == 0:
                                    install_output += f"Successfully installed {pkg} with --no-cache-dir option\n"
                        except Exception as e:
                            install_output += f"Warning: Error installing {pkg}: {str(e)}\n"
                else:
                    install_output = ""
                
                # Create a Python file
                file_path = os.path.join(temp_dir, f"code_{execution_id}.py")
                with open(file_path, 'w') as f:
                    f.write(code)
                
                # Create additional helper files if needed (e.g., for pygame)
                if any(pkg in ['pygame'] for pkg in packages):
                    # Create an assets directory for pygame
                    assets_dir = os.path.join(temp_dir, 'assets')
                    os.makedirs(assets_dir, exist_ok=True)
                
                # Execute the Python code with a timeout
                process = subprocess.run(
                    [python_path, file_path],
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout for safety
                    cwd=temp_dir,
                    env={**os.environ, 'PYTHONPATH': temp_dir}  # Add temp_dir to PYTHONPATH
                )
                
                # Get the result
                result = install_output + process.stdout
                if process.stderr:
                    error = process.stderr
            
            elif language.lower() == 'javascript':
                # Create a JavaScript file
                file_path = os.path.join(temp_dir, f"code_{execution_id}.js")
                with open(file_path, 'w') as f:
                    f.write(code)
                
                # Execute the JavaScript code with Node.js
                process = subprocess.run(
                    ["node", file_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=temp_dir
                )
                
                # Get the result
                result = process.stdout
                if process.stderr:
                    error = process.stderr
            
            elif language.lower() == 'bash' or language.lower() == 'shell':
                # Create a shell script
                file_path = os.path.join(temp_dir, f"code_{execution_id}.sh")
                with open(file_path, 'w') as f:
                    f.write("#!/bin/bash\n" + code)
                
                # Make the script executable
                os.chmod(file_path, 0o755)
                
                # Execute the shell script
                process = subprocess.run(
                    [file_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=temp_dir
                )
                
                # Get the result
                result = process.stdout
                if process.stderr:
                    error = process.stderr
            
            else:
                # Unsupported language
                error = f"Unsupported language: {language}. Supported languages are: python, javascript, bash/shell."
        
        except subprocess.TimeoutExpired:
            error = "Execution timed out after 10 seconds."
        except Exception as e:
            error = f"Error executing code: {str(e)}"
    
    return result, error

def main():
    """Main function to run the Flask app"""
    app.run(host='0.0.0.0', port=7861, debug=True)


if __name__ == "__main__":
    main()