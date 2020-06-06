#!/usr/bin/env bash

. venv/bin/activate
celery worker -A main.celery --loglevel=info --concurrency=1