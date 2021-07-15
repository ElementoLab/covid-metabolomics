.DEFAULT_GOAL := all

NAME=$(shell basename `pwd`)

help:  ## Display help and quit
	@echo Makefile for the $(NAME) project/package.
	@echo Available commands:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		%s\n", $$1, $$2}'

requirements:
	pip install -r requirements.txt

convert:  ## Convert R data to CSV/parquet
	Rscript src/_r_to_parquet.R

analysis:  ## Run Python analysis
	PYTHONPATH=. python -u src/analysis.py

figures:  ## Produce figures in various formats
	cd figures; ./_process.sh

all: convert analysis figures ## Run all steps

.PHONY: figures all
