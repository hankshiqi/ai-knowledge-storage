"""Unified HTTP entry point for the import and query APIs."""
import uvicorn
from fastapi import FastAPI

from app.api.http.import_server import app as import_app
from app.api.http.query_server import app as query_app
from app.infra.config.providers import infra_config


app: FastAPI = query_app
app.mount("/import", import_app, name="import")


if __name__ == "__main__":
    uvicorn.run(app, host=infra_config.settings.app_host, port=infra_config.settings.api_port)
