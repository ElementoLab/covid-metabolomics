# Metabolic measurements of serum from COVID-19 patients

<!-- [![PEP compatible](http://pepkit.github.io/img/PEP-compatible-green.svg)](http://pep.databio.org/) -->


## Organization

- The [metadata](metadata) directory contains metadata relevant to annotate the samples.
- The [src](src) directory contains source code used to analyze the data.
- Raw data will be under the `data` directory.
- Outputs from the analysis will be present in a `results` directory, with subfolders pertaining to each part of the analysis as described below.


## Reproducibility

### Running

To see all available steps type:
```bash
$ make help
```
```
Makefile for the covid-metabolomics project/package.
Available commands:
help            Display help and quit
requirements    Install Python requirements
convert			[dev] Convert R data to CSV/parquet
analysis		Run project analysis
```

To reproduce analysis, simply do:

```bash
$ make requirements
$ make
```

### Requirements

- Python 3.7+ (running on 3.8.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.


### Virtual environment

It is recommended to compartimentalize the analysis software from the system's using virtual environments, for example.

Here's how to create one with the repository and installed requirements:

```bash
git clone git@github.com:ElementoLab/covid-metabolomics.git
cd covid-metabolomics
virtualenv .venv
source activate .venv/bin/activate
pip install -r requirements.txt
```
