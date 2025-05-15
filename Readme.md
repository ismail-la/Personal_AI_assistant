
# Personal AI Assistant

A sophisticated voice-controlled AI assistant that combines speech recognition, text-to-speech, and custom fine-tuned language models to provide an intelligent conversational interface.

## Features

- **Speech Recognition**: Uses [Vosk](https://alphacephei.com/vosk/) for accurate offline speech-to-text conversion
- **Natural Voice Response**: Text-to-speech capabilities for natural-sounding replies
- **Custom Language Models**: Fine-tuned with LoRA adapters for specialized knowledge
- **Expandable Architecture**: Modular design with agent-based reasoning
- **FastAPI Backend**: Modern, high-performance API endpoint

## Technologies

- **Python**: Core programming language
- **Vosk**: Open-source speech recognition
- **Transformers**: Hugging Face models with [PEFT](https://github.com/huggingface/peft) fine-tuning
- **LangChain**: Framework for connecting LLMs with tools and agents
- **FastAPI**: API framework for backend services
- **SoundDevice/SoundFile**: Audio processing libraries

## Project Structure

personal_ai_assistant/
├── agents/ # Agent implementations
│ ├── react_agent.py # ReAct pattern agent for reasoning
│ ├── supervisor.py # Supervision layer
│ └── graph.py # Agent orchestration
├── api/ # API endpoints
│ └── main.py # FastAPI application
├── fine_tune/ # Model fine-tuning
│ ├── data/ # Training datasets
│ └── train_adapter.py # LoRA adapter training script
├── voice_interface/ # Voice interaction components
│ ├── stt.py # Speech-to-text using Vosk
│ ├── tts.py # Text-to-speech conversion
│ └── voice_app.py # Voice application entry point
└── utils/ # Shared utilities
└── config.py # Configuration settings

Feel free to customize this description to better match your project's specific features and goals!

## Setup and Installation

1. **Clone and Install Dependencies**
   ```bash
   # Clone the repository
   git clone https://github.com/ismail-la/personal_ai_assistant.git
   cd personal_ai_assistant

   # Create & activate virtual environment
   python -m venv venv  # Create a virtual environment
   # On Windows:
   venv\Scripts\activate  # Activate on Windows
   # On macOS/Linux:
   source venv/bin/activate  # Activate on macOS/Linux

   # Install Python dependencies
   pip install --no-cache-dir -r requirements.txt


personal_ai_assistant/
├── agents/                     # Agent implementations
│   ├── react_agent.py          # ReAct pattern agent for reasoning
│   ├── supervisor.py           # Supervision layer
│   └── graph.py                # Agent orchestration
├── api/                        # API endpoints
│   └── main.py                 # FastAPI application
├── fine_tune/                  # Model fine-tuning
│   ├── data/                   # Training datasets
│   └── train_adapter.py        # LoRA adapter training script
├── voice_interface/            # Voice interaction components
│   ├── stt.py                  # Speech-to-text using Vosk
│   ├── tts.py                  # Text-to-speech conversion
│   └── voice_app.py            # Voice application entry point
└── utils/                      # Shared utilities
    └── config.py               # Configuration settings
```

## Setup and Installation

1. **Clone and Install Dependencies**
   ```bash
   # Clone the repository
   git clone https://github.com/ismail-la/personal_ai_assistant.git
   cd personal_ai_assistant

   # Create & activate virtual environment
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install Python dependencies
   pip install --no-cache-dir -r requirements.txt
   ```

1. **Download Vosk Model**
   - Download a model from [Vosk models](https://alphacephei.com/vosk/models) (recommended: vosk-model-small-en-us-0.15)
   - Extract to `voice_interface/model/`

## Testing and Usage

### 1. (Optional) Fine-Tune Your Adapter

If you want to train a custom LoRA adapter on your own data:

1. **Prepare training data**
   Create a `fine_tune/data/train.jsonl` file with examples:
   ```json
   {"instruction": "Translate to French: Hello", "response": "Bonjour"}
   {"instruction": "What is machine learning?", "response": "Machine learning is a branch of AI..."}
   ```

2. **Run the trainer**
   ```bash
   python fine_tune/train_adapter.py --base_model gpt2 --data fine_tune/data/train.jsonl
   ```

3. **After training**, your adapter will be saved in `fine_tune/output/`

### 2. Start the FastAPI Server

```bash
# From project root, with venv active
uvicorn api.main:app --reload
```

- Server runs on port 8000 by default
- Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the API using Swagger UI

### 3. Test the Chat Endpoint

#### a) Using cURL
```bash
curl -X POST http://127.0.0.1:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?"}'
```

Expected response:
```json
{"response":"Paris is the capital of France."}
```

#### b) Using Python
Create a file named `test_chat.py`:
```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/chat/",
    json={"prompt": "Summarize the benefits of AI."}
)
print(resp.json()["response"])
```

Run it:
```bash
python test_chat.py
```

### 4. Test Agent Classes Directly

To test your agents in a Python REPL without using the API:

```python
from agents.agent_base import AgentConfig
from agents.supervisor import SupervisorAgent

cfg = AgentConfig("gpt2", "fine_tune/output")
agent = SupervisorAgent(cfg.model_name, cfg.adapter_path)
print(agent.run("List three uses of LangChain."))
```

### 5. Voice Interface

1. Start the API server in one terminal
2. In another terminal, run the voice app:
   ```bash
   python voice_interface/voice_app.py
   ```
3. Speak when prompted—your assistant will transcribe, respond, and speak back

### 6. (Optional) MCP Server for IDE Integration

To let your IDE's AI agents inspect and call your code:

```bash
export FLASK_APP=mcp_server.py
flask run --port 6000
```

Endpoints:
- `POST /read` with `{ "path": "agents/react_agent.py" }`
- `POST /run_chat` with `{ "prompt": "Hello" }`

## Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t personal-ai-assistant .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 personal-ai-assistant
   ```

3. **Test the API** using cURL or your Python script on `localhost:8000`

## Future Improvements

- [ ] More advanced reasoning capabilities
- [ ] Multilingual support
- [ ] Knowledge retrieval from external sources
- [ ] Custom voice model for more natural speech
- [ ] Web interface for configuration
