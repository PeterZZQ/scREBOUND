# method 1
# library("GO.db")
# library("org.Hs.eg.db")
# 
# go_id <- "GO:0008150"
# genes <- select(org.Hs.eg.db, keys = go_id, columns = "ENTREZID", keytype = "GOID")
# gene_names <- mapIds(org.Hs.eg.db, keys = genes$ENTREZID, column = "SYMBOL", keytype = "ENTREZID")

# method 2
rm(list = ls())
gc()
setwd("/project/zzhang834/LLM_KD/batch_encoding")
library(biomaRt)
ensembl <- useEnsembl(biomart = "ensembl", dataset = "hsapiens_gene_ensembl") # For human
go_id <- "GO:0005840"
ribosomal_genes <- getBM(attributes = c("ensembl_gene_id", "external_gene_name"),
            filters = "go",
            values = go_id,
            mart = ensembl)
write.table(as.data.frame(ribosomal_genes), file = "gene_ribosomal_go0005840.csv", quote = FALSE)

go_id <- "GO:0033554"
stress_genes <- getBM(attributes = c("ensembl_gene_id", "external_gene_name"),
                         filters = "go",
                         values = go_id,
                         mart = ensembl)
write.table(as.data.frame(stress_genes), file = "gene_stress_go0033554.csv", quote = FALSE)

go_id <- "GO:0006950"
stress_genes <- getBM(attributes = c("ensembl_gene_id", "external_gene_name"),
                      filters = "go",
                      values = go_id,
                      mart = ensembl)
write.table(as.data.frame(stress_genes), file = "gene_stress_go0006950.csv", quote = FALSE)