#!/usr/bin/env bash

. venv/bin/activate
cd ./backend
celery worker -A main.celery --loglevel=info --concurrency=1