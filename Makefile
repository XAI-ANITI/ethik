install:
	pip install -r requirements.txt

install_dev: install
	pip install -r requirements-dev.txt

test_nb:
	pytest --nbval-lax notebooks

test_unit:
	pytest tests	

test_syntax:
	black --check ethik tests

test: test_nb test_unit test_syntax

doc:
	pdoc --html -f -o docs ethik
