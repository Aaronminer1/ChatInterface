# Changelog

All notable changes to the Chat Interface project will be documented in this file.

## [1.2.0] - 2025-03-28

### Added
- Code execution capabilities for Python, JavaScript, and Bash/Shell
- Split-panel interface with chat on one side and code playground on the other
- Auto-fix toggle that automatically sends errors to the model for correction
- Auto-run toggle that automatically executes code placed in the playground
- Enhanced dependency management for Python code execution
- Clear button for code playground
- Improved code editor colors for better visibility in dark mode

### Technical Details

#### Backend Implementation
- Added code execution environment with sandbox isolation
- Implemented automatic dependency detection and installation
- Created virtual environment management for code execution
- Added support for explicit dependency specification via comments
- Enhanced error handling and feedback loop for code execution
- Increased timeouts for better handling of complex code

#### Frontend Implementation
- Added split-panel layout for chat and code playground
- Implemented toggle switches for auto-fix and auto-run features
- Added clear button for code playground
- Enhanced CSS for better code visibility in dark mode

## [1.1.0] - 2025-03-28

### Added
- Model selection dropdown to switch between different Ollama models
- Model identity awareness to ensure each model correctly identifies itself
- Visual indicators showing which model generated each response
- API endpoint for retrieving available models
- Current model indicator in the UI header
- System messages to inform models of their identity
- Comprehensive model identity mapping for popular models

### Technical Details

#### Backend Implementation
- Added model identity mapping in the OllamaAgent class
- Implemented system message injection for model identity awareness
- Added API endpoint for retrieving available models
- Enhanced model selection handling with validation
- Added dynamic name extraction from model identifiers

#### Frontend Implementation
- Added model selector dropdown in the header
- Implemented model tag display for each response
- Added current model indicator in the UI
- Enhanced message handling to preserve model tags during streaming
- Added visual feedback when changing models

## [1.0.0] - 2025-03-28

### Added
- Initial release of the Simple Agent Interface
- Flask-based web interface with dark mode support
- Ollama integration with gemma3:latest model
- Streaming responses using Server-Sent Events (SSE)
- Markdown rendering with marked.js
- Code syntax highlighting with highlight.js
- Dark mode toggle with system preference detection
- Persistent theme settings using localStorage
- Proper message ordering and conversation flow
- Unique IDs for each response to prevent overwriting
- Welcome message on chat clear
- Complete documentation in README.md

### Technical Details

#### Backend Implementation
- Created Flask application with proper routing
- Implemented Ollama API integration for chat functionality
- Added streaming support using Server-Sent Events
- Created API endpoints for chat communication
- Added environment variable configuration support

#### Frontend Implementation
- Developed responsive UI with CSS variables for theming
- Implemented dark mode with smooth transitions
- Added JavaScript for real-time message updating
- Integrated marked.js for Markdown parsing
- Added highlight.js for code syntax highlighting
- Implemented proper message ordering and layout
- Added unique IDs for each response
- Created theme toggle with system preference detection

#### Bug Fixes
- Fixed message ordering issues
- Resolved duplicate message placeholders
- Fixed response overwriting problem
- Corrected CSS styling for proper message display
- Improved error handling for streaming connections

## Future Plans

See the README.md file for planned future enhancements based on the project roadmap.