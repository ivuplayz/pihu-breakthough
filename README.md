# Pihu-BreakThough

> **Stage 1 Core Backend Foundation**  
> *A clean, modular, production-ready Python & Flask backend service.*

---

## 1. Overview & Stage 1 Scope

**Pihu-BreakThough** is an intelligent assistant backend designed with a modular architecture. Stage 1 establishes the foundational core backend service using Python, Flask, and an application-factory pattern.

### Key Stage 1 Principles:
- **Zero Required External Dependencies for Local Run**: Can be executed immediately without external API keys, third-party accounts, active database clusters, or complex local OS dependencies.
- **Strict Separation of Concerns**: Clean modular design separating configuration, API blueprints, database management, LLM provider interfaces, and request routing.
- **Fail-Safe Database Abstraction**: Integrates `psycopg 3` and Neon PostgreSQL connection abstractions that cleanly report `database: "NOT CONFIGURED"` when `DATABASE_URL` is absent, preventing crashes or startup blocking.
- **Deterministic Stage 1 Provider**: Serves deterministic, structured responses confirming operational status while exposing a future-compatible response schema.
- **Enterprise-Grade Validation & Security**: Rigorous JSON parsing, input sanitization, length boundaries, role validation, and production error handlers that never leak Python stack traces.
- **Cloud & Serverless Ready**: Includes standard `vercel.json` deployment rules and WSGI compliance.

---

## 2. Project Architecture & Directory Layout

```
pihu-breakthough-1/
├── app.py                      # Application entry point & factory function create_app()
├── requirements.txt            # Minimal runtime dependencies (Flask, python-dotenv, psycopg)
├── vercel.json                 # Vercel deployment configuration for Flask WSGI
├── .env.example                # Template listing all current and future environment variables
├── .gitignore                  # Python, environment, and build ignore rules
├── README.md                   # Comprehensive project documentation
├── api/                        # HTTP API presentation layer (Blueprints)
│   ├── __init__.py             # API Blueprint definition (/api prefix)
│   ├── health.py               # GET /api/health endpoint
│   └── chat.py                 # POST /api/chat endpoint with strict validation
├── pihu_core/                  # Core business logic & provider abstractions
│   ├── __init__.py             # Package exports
│   ├── config.py               # Safe environment variable configuration manager
│   ├── database.py             # Neon PostgreSQL / psycopg 3 connection foundation
│   ├── providers.py            # Base provider interface and Stage 1 deterministic provider
│   └── router.py               # Request and provider router with fallback mechanism
└── tests/                      # Automated test suite
    ├── __init__.py             # Test package initialization
    └── test_stage1.py          # Comprehensive test suite validating all endpoints & validation rules
```

---

## 3. Application Factory Pattern

The application is structured using Flask's `create_app()` factory pattern in `app.py`:

```python
from flask import Flask
from api import api_bp
from pihu_core.config import Config

def create_app(config_class: type[Config] = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config["MAX_CONTENT_LENGTH"] = config_class.MAX_CONTENT_LENGTH
    
    app.register_blueprint(api_bp)
    
    # Custom error handlers (400, 404, 405, 413, 500)
    ...
    return app
```

This guarantees:
1. **Isolated Testing**: Tests can instantiate fresh application instances with isolated configurations.
2. **Flexible Deployment**: Supports both WSGI servers (`gunicorn app:app`), serverless runtimes (Vercel), and local CLI execution.

---

## 4. API Reference

### 4.1. Root Service Check
- **Endpoint**: `GET /`
- **Description**: Lightweight health and service identification.
- **Response** `(200 OK)`:
```json
{
  "ok": true,
  "app": "Pihu-BreakThough",
  "service": "brain"
}
```

---

### 4.2. Health Check
- **Endpoint**: `GET /api/health`
- **Description**: Returns detailed system status, active database state, and provider availability.
- **Response** `(200 OK)`:
```json
{
  "status": "healthy",
  "app": "Pihu-BreakThough",
  "stage": "Stage 1",
  "database": "NOT CONFIGURED",
  "router": "active",
  "active_provider": "stage1-deterministic",
  "available_providers": [
    "stage1-deterministic"
  ],
  "timestamp": "2026-09-23T09:30:00.000000+00:00"
}
```
*Note: If `DATABASE_URL` is configured, `database` reports `"CONNECTED"` or `"CONNECTION_FAILED"` safely.*

---

### 4.3. Chat Interaction
- **Endpoint**: `POST /api/chat`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "message": "Hello, Pihu-BreakThough!",
  "history": [
    {
      "role": "user",
      "content": "Hi there"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I assist you today?"
    }
  ],
  "provider": "stage1-deterministic"
}
```

#### Fields:
| Field | Type | Required | Constraints |
| :--- | :--- | :--- | :--- |
| `message` | string | **Yes** | 1 to 4000 non-whitespace characters |
| `history` | array of objects | No | Up to 50 items. Each item must have `role` (`user` \| `assistant` \| `system`) and `content` (string) |
| `provider` | string | No | Optional provider override. Defaults to active provider with fallback |

#### Successful Response `(200 OK)`:
```json
{
  "ok": true,
  "app": "Pihu-BreakThough",
  "response": "Pihu-BreakThough Stage 1 Core is operational. Received message (24 chars): \"Hello, Pihu-BreakThough!\". History context contains 2 items.",
  "provider": "stage1-deterministic",
  "model": "core-deterministic-v1",
  "metadata": {
    "history_length": 2,
    "received_chars": 24,
    "stage": "Stage 1",
    "timestamp": "2026-09-23T09:30:00.000000+00:00"
  }
}
```

---

## 5. Input Validation & Error Handling

All error responses strictly follow a uniform JSON structure. **Python stack traces are never exposed to the client in production.**

### Error Response Format:
```json
{
  "ok": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable description."
  }
}
```

### Supported Validation & Error Codes:
| Error Code | HTTP Status | Trigger Condition |
| :--- | :--- | :--- |
| `INVALID_CONTENT_TYPE` | 400 | Request headers lack `Content-Type: application/json` |
| `INVALID_JSON` | 400 | Malformed JSON body or non-object root payload |
| `MISSING_MESSAGE` | 400 | Payload omitted the required `message` key |
| `INVALID_MESSAGE_TYPE` | 400 | `message` was provided as a non-string type |
| `EMPTY_MESSAGE` | 400 | `message` is empty string or only whitespace |
| `MESSAGE_TOO_LONG` | 400 | `message` exceeds 4000 characters |
| `INVALID_HISTORY` | 400 | `history` is not a list |
| `HISTORY_TOO_LONG` | 400 | `history` contains > 50 messages |
| `INVALID_HISTORY_ITEM` | 400 | History element is not a JSON object |
| `INVALID_HISTORY_ROLE` | 400 | History role is not `user`, `assistant`, or `system` |
| `INVALID_HISTORY_CONTENT` | 400 | History content is not a string |
| `NOT_FOUND` | 404 | Non-existent route requested |
| `METHOD_NOT_ALLOWED` | 405 | Method (e.g. GET instead of POST) not supported |
| `PAYLOAD_TOO_LARGE` | 413 | Request size exceeds `MAX_CONTENT_LENGTH` (1 MB) |
| `INTERNAL_SERVER_ERROR` | 500 | Unhandled internal exception caught gracefully |

---

## 6. Local Setup & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- `pip`

### Step 1: Clone & Navigate
```bash
cd pihu-breakthough-1
```

### Step 2: Create & Activate Virtual Environment (Optional but recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Variables (Optional)
```bash
# Copy example configuration
cp .env.example .env
```
*Note: In Stage 1, `.env` is completely optional. Default parameters operate without any configuration file.*

### Step 5: Start the Server
```bash
python app.py
```
Output:
```
[2026-09-23 15:00:00,000] [INFO] in app: Starting Pihu-BreakThough (Stage 1) on port 5000 [debug=False]
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

---

## 7. Verification & Automated Tests

A dedicated test suite is provided in `tests/test_stage1.py` using standard `unittest`:

```bash
python tests/test_stage1.py
```

### Test Coverage Highlights:
- Validates root diagnostic endpoint `GET /`.
- Validates health check `GET /api/health` without database connection.
- Validates normal chat flow with and without history.
- Validates all 9 input rejection rules for `/api/chat`.
- Validates global 404, 405, and 413 error handlers.

---

## 8. Deployment on Vercel

The included `vercel.json` provides zero-configuration Flask deployment:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

---

## 9. Future Roadmap

- **Stage 2**: Neon PostgreSQL connection pooling, database migrations, message persistence, and session management.
- **Stage 3**: Integration of Google Gemini, Groq, Hugging Face, and local Ollama model providers into `ProviderRouter`.
- **Stage 4**: Authentication, rate limiting, and web frontend / streaming chat client.
