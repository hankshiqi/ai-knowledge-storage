# Unified API entry point

The import and query FastAPI applications can run in one Uvicorn process through `app.api.main:app`.
The query API keeps its existing paths. The import API is mounted under `/import`.

| Capability | Unified URL |
| --- | --- |
| Query UI | `/query/html` |
| Query endpoint | `/query` |
| Query SSE stream | `/stream/{session_id}` |
| Import UI | `/import/html` |
| File upload | `/import/upload` |
| Import task status | `/import/status/{task_id}` |
| Query health check | `/health` |

Start the unified service from the repository root:

```powershell
.venv\Scripts\python.exe -m app.api.main
```

The service listens on `APP_HOST` and `API_PORT` (`0.0.0.0:8000` by default). The equivalent Uvicorn command is:

```powershell
.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

This combines HTTP routing and application lifecycle in one process. It does not restart the process after a crash; production deployment should run it under a process manager such as Docker Compose with a restart policy, systemd, or Windows Service Manager. The individual `import_server.py` and `query_server.py` launch blocks remain useful for isolated local debugging.
