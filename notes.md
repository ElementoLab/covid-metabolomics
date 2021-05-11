
## General

"DL_" features are lipoproteins.


lipoprotein abbreviations:
 - FC: free cholesterol
 - CE: esterified cholesterol
 - PL: phospholipids

 - VLDL: very low density lipoproteins
 - LDL: low density lipoproteins
 - HDL: high density lipoproteins
 - VHDL: very high density lipoproteins


## 2021/02/28
 - Got data from Jan Krumsiek as SummarizedExperiment
 - Started project
 - Converted to CSV and Parquet

## 2021/03/09
 - Unsupervised analysis:
   - Spectral embedding works well, similar to LSA paper
 - Supervised analysis seems to recover several associations with progression/severity and even some specific to some groups e.g. 'severe' patients.

## 2021/04/01
 - Exploration of patient speed of motion through latent spaces


## 2021/04/13 Meeting:

Use clinical lab values -> correlate with clinical

Get feature modules -> correlate with clinical


Check Olink papers.

Additional papers:
https://www-nature-com.ezproxy.med.cornell.edu/articles/s41598-021-82426-7

https://pubmed-ncbi-nlm-nih-gov.ezproxy.med.cornell.edu/33440724/

https://faseb.onlinelibrary.wiley.com/doi/epdf/10.1096/fj.202001451
https://link-springer-com.ezproxy.med.cornell.edu/chapter/10.1007/978-3-319-09665-0_15


## Additional datasets:

Olink:
 - https://www.olink.com/mgh-covid-study/
 - https://info.olink.com/broad-covid-study-overview-download?submissionGuid=b5a527b0-09a7-4f3c-8532-e90674beb4e0
 - https://www.olink.com/category/covid19/



## Data integration

CCA works well with regularization
Impute remaining data (Flow measurements for NMR samples)

Check trajectories in shared space

Check trajectories per patient in Flow cytometry, measure speed
Compare speed measurements between NMR and Flow


## Feature context:
Marielle/Berend's lipid circle paper: https://doi.org/10.1016/j.cell.2015.05.051
Resource for 
https://lipidmaps.org/


### 2021/05/10:
Make network, export to Gephi, plot & export locations.
	Plot in matplotlib in a grid (size vs density)

OR plot in scanpy, force_atlas2 with edges

OR hierarchical clustering
