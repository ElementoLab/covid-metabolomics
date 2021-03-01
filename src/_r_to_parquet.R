#!/usr/bin/env R

# This script reads a Rdata file containing a SummarizedExperiment object
# and exports CSV/Parquet files.
# Requires the SummarizedExperiment library:
# 	`apt install r-bioc-summarizedexperiment`

library("SummarizedExperiment")
library("arrow")

load("data/Elemento_COVID19_precalc.rds")

x <- t(D@assays@data[[1]])
y <- D@colData

write.csv(x, "data/assay_data.csv")
write.csv(y, "data/metadata.csv")

write_parquet(as.data.frame(x), "data/assay_data.pq")
write_parquet(as.data.frame(y), "data/metadata.pq")
