.PHONY: run test lint clean

run:
	python src/run.py "Analyze ROAS drop"

test:
	pytest -q

lint:
	flake8 src || true

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
