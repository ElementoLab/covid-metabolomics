# Metabolic and immune markers for precise monitoring of COVID-19 severity and treatment

Andre Figueiredo Rendeiro,  Charles Kyriakos Vorkas,  Jan Krumsiek, Harjot Singh,  Shashi N Kapadia,  Luca Vincenzo Cappelli,  Maria Teresa Cacciapuoti,  Giorgio Inghirami,  Olivier Elemento,  Mirella Salvatore.
**Metabolic and immune markers for precise monitoring of COVID-19 severity and treatment**. MedRxiv (2021). doi:10.1101/2021.09.05.21263141

[![medRxiv DOI badge](https://zenodo.org/badge/doi/10.1101/2021.09.05.21263141.svg)](https://doi.org/10.1101/2021.09.05.21263141) ⬅️ read the preprint here


## Organization

- The [data](data) directory contains the assay data (NMR).
- The [metadata](metadata) directory contains metadata relevant to annotate the samples.
- The [src](src) directory contains source code used to analyze the data.
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

If you wish to run a portion of the analysis interactively say with IPython, make sure the repository root is added to your `PYTHONPATH` to allow importing of the `src` module. IPython may already do this by default.

### Requirements

- Python 3.8+ (running on 3.8.2)
- Python packages as specified in the [requirements file](requirements.txt) - install with `make requirements` or `pip install -r requirements.txt`.
