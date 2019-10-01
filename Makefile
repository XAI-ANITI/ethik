install:
	pip install -r requirements.txt

install_dev: install
	pip install -r requirements-dev.txt

test_nb:
	pytest --nbval-lax --current-env notebooks

test_unit:
	pytest

test_syntax:
	black --check ethik tests

test: test_nb test_unit test_syntax

update_nb:
	jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb --ExecutePreprocessor.kernel_name=${IPY_KERNEL} --ExecutePreprocessor.timeout=300

nb_to_html: update_nb
	rm -rf docs/notebooks
	mkdir -p docs/notebooks
	jupyter nbconvert --to html --output-dir docs/notebooks notebooks/*.ipynb --template=docs/notebook.tpl

api_ref:
	mkdir -p docs
	pdoc --html -c latex_math=1 -f -o docs ethik
	mv docs/ethik/* docs/
	rm -r docs/ethik/

doc: nb_to_html api_ref

deploy_doc: doc
	git clone -b gh-pages "https://${GH_USER}:${GH_PASSWORD}@github.com/${TRAVIS_REPO_SLUG}.git" gh-pages
	cp -R docs/* gh-pages/
	cd gh-pages
	git add *
	git diff --staged --quiet && echo "$0: No changes to commit." && exit 0
	git commit -a -m "CI: Update docs for ${TRAVIS_TAG} (${head})"
	git push
