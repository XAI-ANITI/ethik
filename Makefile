install:
	pip install -r requirements.txt

install_dev: install
	pip install -r requirements-dev.txt

test_nb:
	pytest --nbval-lax notebooks

test_unit:
	pytest tests	

test: test_nb test_unit

doc:
	pdoc --html -f -o docs ethik
