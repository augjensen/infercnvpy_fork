# CNV Inference Method

This document describes the CNV inference method implemented in `infercnvpy`.

## Overview

The CNV inference method in `infercnvpy` is designed to infer Copy Number Variations (CNVs) from single-cell RNA sequencing (scRNA-seq) data. The method is based on the principle that the average gene expression over a genomic region is proportional to the copy number of that region.

## Input Data

The CNV inference method requires the following input data:

*   **Expression Data:** An `AnnData` object with a gene expression matrix in `adata.X` or a specified `layer`, where the rows are cells and the columns are genes. The expression data should be normalized and log-transformed.
*   **Genomic Position Data:** The `adata.var` dataframe must contain `chromosome`, `start`, and `end` columns with the genomic position of each gene.
*   **Breakpoint Data (Optional):** A CSV file with breakpoint data. If provided via the `breakpoint_file` argument in `infercnv.tl.infercnv`, the CNV inference will be based on the segments defined in this file. The file should have the following columns: `seg_chr`, `seg_start`, `seg_end`. A `segment_id` column is optional and will be automatically generated if not provided.

## CNV Inference without Breakpoints

When no breakpoint data is provided, the CNV inference is performed using a running window approach. The method consists of the following steps:

1.  **Log-fold Change Calculation:** The log-fold change of each gene is calculated by subtracting the average expression of the reference cells from the expression of each cell.
2.  **Clipping:** The log-fold change values are clipped to a certain range to reduce the effect of outliers.
3.  **Smoothing:** The clipped log-fold change values are smoothed by genomic position using a running window. The size of the window can be adjusted by the user.
4.  **Centering:** The smoothed values are centered by subtracting the median value of each cell.
5.  **Noise Filtering:** A dynamic threshold is applied to filter out noise.

## CNV Inference with Breakpoints

When breakpoint data is provided, the CNV inference is performed using a segment-based approach. The method consists of the following steps:

1.  **Log-fold Change Calculation:** The log-fold change of each gene is calculated by subtracting the average expression of the reference cells from the expression of each cell.
2.  **Clipping:** The log-fold change values are clipped to a certain range to reduce the effect of outliers.
3.  **Segment-based Averaging:** The clipped log-fold change values are averaged over each genomic segment defined by the breakpoints.
4.  **Handling of low-gene segments:** Segments with fewer genes than `min_genes_per_segment` (a parameter in `infercnv.tl.infercnv`) will be masked and will not be used for CNV inference. Their value will be set to `NaN`.
5.  **Centering:** The segment-based average values are centered by subtracting the median value of each cell.
6.  **Noise Filtering:** A dynamic threshold is applied to filter out noise.

## Visualization

The results of the CNV inference can be visualized using the `infercnvpy.pl.chromosome_heatmap` function.

### Heatmap with Breakpoints

When CNV inference was performed using breakpoints, the heatmap has the following features:

*   **Greyed-out areas:** Segments that were masked during the CNV inference because they contained too few genes are shown in grey. This allows to visually identify regions where the CNV inference is not reliable.
*   **Segment annotations:** The locations of the genomic segments can be displayed at the bottom of the heatmap, below the chromosome annotations. This helps to relate the CNV calls to the underlying genomic structure.

## Algorithm Details

The CNV score for each segment is calculated as follows:

1.  **Calculate the average log-fold change for each segment:**

    ```
    LFC_seg = mean(LFC_gene1, LFC_gene2, ...)
    ```

    where `LFC_gene` is the log-fold change of a gene in the segment. If the number of genes in a segment is below `min_genes_per_segment`, `LFC_seg` will be `NaN`.

2.  **Center the segment log-fold change values:**

    ```
    LFC_seg_centered = LFC_seg - median(LFC_seg_cell1, LFC_seg_cell2, ...)
    ```

3.  **Apply noise filtering:**

    ```
    if abs(LFC_seg_centered) < dynamic_threshold * std(LFC_seg_centered):
        CNV_score = 0
    else:
        CNV_score = LFC_seg_centered
    ```