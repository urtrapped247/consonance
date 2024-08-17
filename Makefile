.DEFAULT_GOAL := default

run_generate:
	python -c 'from consonance.generate import generate_synthetic_musicxml; generate_synthetic_musicxml()'

reinstall_package:
	@pip uninstall -y consonance || :
	@pip install -e .

run_api:
	uvicorn consonance.api.fast:app --reload