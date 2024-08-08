.DEFAULT_GOAL := default

run_generate:
	python -c 'from consonance.generate import generate_synthetic_musicxml; generate_synthetic_musicxml()'
