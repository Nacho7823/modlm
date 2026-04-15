# modllm Chat

Chat LLM con Vite (frontend) + FastAPI + LangChain (backend).

## Estructura

```
modllm/
├── backend/           # FastAPI + LangChain (Python)
│   ├── app/
│   │   ├── api/     # Endpoints REST
│   │   ├── core/    # Config, exceptions
│   │   ├── domain/  # Pydantic schemas
│   │   └── services/ # LLM, storage
│   └── .env
├── frontend/         # Vite + React + TypeScript
│   └── src/
│       ├── components/
│       ├── hooks/
│       ├── types/   # message, chat, history, tool, config
│       └── utils/   # api, formatters, errors
└── shared/          # Tipos compartidos
```

## Requisitos

- Python 3.10+
- Node.js 18+
- API key de OpenAI (u otro proveedor compatible)

## Instalación

### Backend

```bash
cd backend
uv venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
uv pip install -r requirements.txt
```

Configurar `.env`:
```bash
OPENAI_API_KEY=sk-...
```

### Frontend

```bash
cd frontend
npm install
```

## Ejecutar

### Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm run dev
```

Acceder a `http://localhost:5173`

## API Endpoints

| Método | Path | Descripción |
|--------|------|-------------|
| POST | `/api/chat` | Enviar mensaje |
| GET | `/api/chat/history` | Obtener historial |
| GET | `/api/chat/{chat_id}` | Obtener chat |
| DELETE | `/api/chat/{chat_id}` | Eliminar chat |
| DELETE | `/api/chat` | Limpiar historial |
| GET | `/api/config` | Obtener config LLM |
| PUT | `/api/config` | Actualizar config LLM |

## Clean Code

El proyecto sigue los principios de Clean Code:
- Nombres significativos
- Funciones pequeñas (SRP)
- Tipos TypeScript/Python divisionados
- Manejo de errores completo
- Tests unitarios (estructura)

## Tecnología

| Capa | Stack |
|------|-------|
| Frontend | Vite, React, TypeScript |
| Backend | FastAPI, Pydantic, LangChain |
| Storage | JSON file |
| LLM | LangChain (OpenAI compatible) |