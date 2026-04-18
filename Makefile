install:
	pip install -r requirements.txt

test:
	pytest tests/

run:
	python experiments/run_experiments.py

lint:
	flake8 src/

format:
	black src/

notebook:
	jupyter notebook