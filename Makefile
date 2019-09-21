install:
	pip install -r requirements.txt

install_dev: install
	pip install -r requirements-dev.txt

test_nb:
	pytest --nbval-lax notebooks

test_unit:
	pytest --doctest-modules

test_syntax:
	black --check ethik tests

test: test_nb test_unit test_syntax

nb_to_html:
	mkdir -p docs/notebooks
	jupyter nbconvert --to html --output-dir docs/notebooks notebooks/*.ipynb

doc: nb_to_html
	pdoc --html -f -o docs ethik
