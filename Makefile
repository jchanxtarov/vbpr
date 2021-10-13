default: | help

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "   run              to run using sample dataset with saving all files"
	@echo "   dry-run          to run using sample dataset without saving any files"
	@echo "   check            to type check"
	@echo "   setup            to setup to run"
	@echo "   args             to see argments"

# run using sample dataset with saving any logfiles
run:
	poetry run python src/main.py --save_log --save_model

# run using sample dataset without saving any files
dry-run:
	poetry run python src/main.py

# type check
check:
	poetry run mypy src/main.py

# setup to run
setup:
	poetry install

# see argments
args:
	poetry run python src/main.py --help
