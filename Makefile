.PHONY: env fmt check test

env:
	if [ ! -d "env" ]; then \
		python -m venv env; \
	fi
	. ./env/bin/activate && pip install -r requirements.txt && pip install -e .

fmt:
	ruff check --select I --fix
	ruff format

check:
	ruff check
	pyright .

test:
	pytest test/*
