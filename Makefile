# This file is for development usage

SHELL=/bin/bash

.PHONY:reinstall
reinstall:
	rm -f ./*.so
	pip uninstall torch_memory_saver -y
	# pip install --no-cache-dir -e .
	pip install --no-cache-dir .

.PHONY:clean
clean:
	rm -rf dist/*

.PHONY:build
build:
	python3 -m build --no-isolation

.PHONY:upload
upload:
	python3 -m twine upload dist/*
