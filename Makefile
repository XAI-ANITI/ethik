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

update_nb:
	jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb

nb_to_html: update_nb
	mkdir -p docs/notebooks
	jupyter nbconvert --to html --output-dir docs/notebooks notebooks/*.ipynb

api_ref:
	pdoc --html -c latex_math=1 -f -o docs ethik

doc: nb_to_html api_ref
