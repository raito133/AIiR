#!/usr/bin/env bash

. venv/bin/activate
celery worker -A server.modules.tasks.celery --loglevel=info --concurrency=1