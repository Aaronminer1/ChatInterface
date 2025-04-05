# Chat Interface with Auto-Run and Auto-Fix Features

A modern, feature-rich agent interface that uses Ollama with multiple language models. This project provides a web-based UI where you can chat with different models in real-time with streaming responses, execute code, and automatically fix errors.

## Features

- **Modern Web Interface**: Clean, responsive design with dark mode support
- **Streaming Responses**: See the model's responses as they're being generated
- **Markdown Rendering**: Full support for Markdown formatting with syntax highlighting
- **Local LLM Integration**: Uses Ollama for local inference with multiple models
- **Model Selection**: Switch between different Ollama models at runtime
- **Model Identity Awareness**: Each model correctly identifies itself when asked
- **Conversation History**: Maintains complete chat history with proper message ordering
- **Dark Mode Toggle**: Switch between light and dark themes for better readability

### New Features

- **Code Execution**: Execute Python, JavaScript, and Bash/Shell code directly in the interface
- **Auto-Run Toggle**: Automatically execute code placed in the playground
- **Auto-Fix Toggle**: Automatically send errors to the model for correction
- **Advanced Dependency Management**: Detect and install required packages for Python code
- **Split-Panel Interface**: Chat on one side, code playground on the other
- **Clear Button**: Easily reset the code playground
- **Improved Dark Mode**: Enhanced code editor colors for better visibility

## Project Structure

```
/
├── agent.py           # Main Flask application and Ollama agent
├── requirements.txt   # Python dependencies
├── templates/         # HTML templates
│   └── index.html     # Main chat interface
├── README.md          # Project documentation
├── CHANGELOG.md       # Version history
└── TECHNICAL.md       # Technical documentation
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Aaronminer1/ChatInterface.git
   cd ChatInterface
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Make sure you have Ollama installed and running:
   - [Ollama Installation Instructions](https://github.com/ollama/ollama)

4. Run the application:
   ```bash
   python agent.py
   ```

5. Open your browser and navigate to `http://localhost:7861`

## Usage

- Select a model from the dropdown
- Type a message and click "Send" or press Enter
- Toggle dark mode with the switch in the top right
- Use the code playground to execute code
- Toggle auto-fix and auto-run features as needed

## Configuration

You can configure the application by setting environment variables:

- `OLLAMA_BASE_URL`: URL of your Ollama instance (default: http://localhost:11434)
- `MODEL_NAME`: Default model to use (default: gemma3:latest)

## License

This project is open source and available under the Apache-2.0 license.

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama) for providing the local LLM inference capability
