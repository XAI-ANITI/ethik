install:
	python setup.py install

install_dev: install
	pip install -r requirements-dev.txt
	pre-commit install # For black

test_nb:
	pytest --nbval-lax --current-env notebooks

test_unit:
	pytest

test_syntax:
	black --check ethik tests

test: test_syntax test_unit  test_nb

update_nb:
	jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb --ExecutePreprocessor.kernel_name=${IPY_KERNEL} --ExecutePreprocessor.timeout=300

nb_to_html:
	rm -rf docs/tutorials
	mkdir -p docs/tutorials
	jupyter nbconvert --to html --output-dir docs/tutorials notebooks/*.ipynb --template=docs/notebook.tpl

tutorials: update_nb nb_to_html
	cp docs/tutorials.html docs/tutorials/index.html

api_ref:
	rm -rf docs/api
	mkdir -p docs/api
	# We need to disable minifying so that the indentation in Jekyll's front matter
	# is not altered
	pdoc --html --template-dir docs/api_templates/ -c latex_math=1 -c minify=0 -f -o docs ethik
	mv docs/ethik/* docs/api/
	rm -r docs/ethik/

gallery:
	python docs/create_gallery.py

doc: tutorials api_ref gallery

deploy_doc: doc
	git clone -b gh-pages "https://${GH_USER}:${GH_PASSWORD}@github.com/${TRAVIS_REPO_SLUG}.git" gh-pages
	cp -R docs/* gh-pages/
	# All the commands need to be on the same line to stay in the "gh-pages" folder
	cd gh-pages; git add *; \
	  git diff --staged --quiet && echo "$0: No changes to commit." && exit 0; \
	  git commit -a -m "CI: Update docs for ${TRAVIS_TAG} (${head})"; \
	  git push

serve_website:
	cd docs; bundle exec jekyll serve

install_website:
	gem install jekyll bundler
	bundle install
	bundle update
