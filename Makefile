.DEFAULT_GOAL := default

run_generate:
	python -c 'from taxifare.interface.main import train; train()'