# Technical Documentation

This document provides detailed technical information about the Chat Interface implementation.

## Architecture Overview

The Chat Interface follows a client-server architecture:

1. **Server (Backend)**: A Flask application that serves the web interface and handles API requests to Ollama
2. **Client (Frontend)**: A web interface built with HTML, CSS, and JavaScript that communicates with the server
3. **LLM Service**: Ollama running locally, providing access to various language models
4. **Code Execution Environment**: Sandboxed environment for executing code with dependency management

```
[User Browser] <---> [Flask Server] <---> [Ollama API] <---> [Language Models]
                         |
                         v
               [Code Execution Environment]
```

## Backend Implementation (agent.py)

### Key Components

1. **OllamaAgent Class**:
   - Handles communication with the Ollama API
   - Provides methods for chat and streaming chat
   - Manages message history and formatting

2. **Flask Application**:
   - Serves the web interface (index.html)
   - Provides API endpoints for chat communication
   - Handles streaming responses using Server-Sent Events (SSE)

3. **Environment Configuration**:
   - Uses python-dotenv for configuration management
   - Allows customization of Ollama URL and model name

### API Endpoints

- **GET `/`**: Serves the main chat interface
- **POST `/api/chat`**: Handles chat messages with support for both streaming and non-streaming modes
- **POST `/api/execute`**: Executes code in a sandboxed environment and returns the result
- **GET `/api/models`**: Returns a list of available models from Ollama

### Streaming Implementation

The streaming implementation uses Server-Sent Events (SSE) to send partial responses to the client as they are generated:

1. The client sends a POST request to `/api/chat` with the message and chat history
2. The server initiates a streaming request to Ollama
3. As chunks of the response are received from Ollama, they are forwarded to the client
4. The client updates the UI in real-time as chunks arrive

### Code Execution Environment

The code execution environment is designed to safely run code submitted by users while providing helpful features like dependency management:

1. **Sandboxed Execution**:
   - Code is executed in isolated temporary directories
   - Each execution has a unique ID to prevent conflicts
   - Resource limits are enforced through timeouts (30 seconds for execution)
   - Virtual environments are created for Python code execution

2. **Dependency Management**:
   - **Automatic Detection**: Parses import statements to identify required packages
   - **Explicit Specification**: Supports `# pip install package_name` comments
   - **Package Mapping**: Recognizes common aliases (e.g., `np` → `numpy`, `plt` → `matplotlib`)
   - **Installation Process**:
     - Creates isolated virtual environments for each execution
     - Upgrades pip to ensure compatibility
     - Installs packages with timeout protection
     - Provides fallback installation with `--no-cache-dir` option

3. **Language Support**:
   - **Python**: Full support with dependency management
   - **JavaScript**: Basic execution support
   - **Bash/Shell**: Command execution with safety constraints

4. **Error Handling and Feedback**:
   - Captures both stdout and stderr
   - Provides detailed error messages
   - Supports automatic error correction through model feedback
   - Maintains context for error-fixing requests

### Auto-Fix and Auto-Run Features

1. **Auto-Fix Implementation**:
   - Detects execution errors in user or model-generated code
   - Sends error context back to the model with a request to fix
   - Applies the fixed code to the playground
   - Can be toggled on/off through the UI

2. **Auto-Run Implementation**:
   - Automatically executes code when it's placed in the playground
   - Integrates with the Auto-Fix feature for seamless error correction
   - Can be toggled on/off through the UI

## Frontend Implementation (index.html)

### Key Components

1. **UI Elements**:
   - Chat container for displaying messages
   - Message input field and send button
   - Theme toggle for switching between light and dark modes
   - Clear chat button

2. **JavaScript Functionality**:
   - Handles user input and message sending
   - Processes streaming responses
   - Updates the UI in real-time
   - Manages theme switching and persistence
   - Renders Markdown content with syntax highlighting

3. **CSS Styling**:
   - Uses CSS variables for theming
   - Provides responsive design for different screen sizes
   - Implements proper message bubbles and layout
   - Supports both light and dark modes

### Theme Implementation

The theme system uses CSS variables and JavaScript to provide a seamless theme switching experience:

1. CSS variables define colors for both light and dark modes
2. JavaScript detects the system preference and applies the appropriate theme
3. Theme preference is saved in localStorage for persistence
4. A toggle switch allows manual switching between themes

### Message Handling

Messages are handled with a sophisticated system to ensure proper ordering and display:

1. Each message is wrapped in a container for proper layout
2. User messages appear on the right with a different style
3. Assistant messages appear on the left with Markdown rendering
4. Each response has a unique ID to prevent overwriting
5. Streaming updates are applied to the correct message element

## LLM Integration

The integration with Ollama provides access to multiple models:

1. The OllamaAgent class handles communication with the Ollama API
2. Requests are sent to the `/api/chat` endpoint of Ollama
3. Streaming is enabled for real-time response generation
4. Message history is maintained for context in conversations
5. Model selection is available through a dropdown interface
6. Each model maintains its proper identity through system messages

## Security Considerations

1. The application is designed for local use only and should not be exposed to the internet
2. No authentication is implemented as it's intended for personal use
3. Input validation is minimal and should be enhanced for production use
4. The application does not store any data persistently except for theme preference

## Performance Considerations

1. Streaming responses reduce perceived latency for users
2. The application is lightweight and should run well on most systems
3. Memory usage is minimal as no large datasets are loaded
4. The performance is primarily limited by the speed of the Ollama model inference

## Implemented Enhancements

### Model Selection

The application now supports selecting different Ollama models at runtime:

1. A dropdown menu in the UI allows switching between available models
2. The backend fetches the list of available models from Ollama
3. The selected model is used for subsequent messages
4. Each message displays which model generated it
5. The UI updates to show the currently active model

### Model Identity Awareness

Each model now correctly identifies itself when asked about its identity:

1. A mapping of model names to their proper identities is maintained
2. System messages inform each model about its true identity
3. Models respond accurately when asked about who created them
4. Model names are properly extracted and formatted
5. This prevents models from incorrectly identifying themselves

## Future Technical Enhancements

Based on the project memories, several additional enhancements are planned:

1. **Plugin Architecture**:
   - Standardized tool integration interface
   - Dynamic tool loading and dependency management
   - Tool composition for complex tasks

2. **Enhanced Reasoning**:
   - Chain-of-thought planning for better transparency
   - Knowledge graph integration
   - Advanced task decomposition