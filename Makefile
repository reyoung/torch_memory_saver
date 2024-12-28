# This file is for development usage

SHELL=/bin/bash

.PHONY:reinstall
reinstall:
	rm -f ./*.so
	pip uninstall torch_memory_saver -y
	pip install --no-cache-dir -e .
