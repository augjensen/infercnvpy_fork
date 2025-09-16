# Project: Incorporate scDNA-seq Breakpoints into infercnvpy

This document outlines the plan to modify `infercnvpy` to use scDNA-seq breakpoint data for more accurate CNV inference.

## Project Goal

The primary goal is to extend `infercnvpy` to accept scDNA-seq breakpoint information (which genes belong to which segments) and use it to refine the CNV inference process. This will allow for more accurate and biologically relevant CNV calls, especially in complex cancer datasets.

## Proposed Changes

1.  **Input Data:**
    *   A new file format will be defined to input the breakpoint data. This could be a simple TSV or CSV file with columns for `chromosome`, `start`, `end`, and `segment_id`.
    *   The `infercnvpy.io.read()` function will be updated to handle this new file format.

2.  **CNV Inference Algorithm:**
    *   The core CNV inference algorithm in `infercnvpy.infercnv` will be modified to use the breakpoint information.
    *   The current approach of using a sliding window or chromosome-wide averaging will be replaced with a segment-based approach. For each segment defined by the breakpoints, the average expression of the genes within that segment will be used to determine the CNV state.

3.  **Visualization:**
    *   The plotting functions in `infercnvpy.pl` will be updated to visualize the CNV calls based on the new segment-based approach. This will likely involve modifying the heatmap plotting function to draw segment boundaries.

## Step-by-Step Guide

1.  **Implement Breakpoint File Reader:**
    *   Create a new function in `infercnvpy.io` to read the breakpoint file and store the data in a suitable data structure (e.g., a pandas DataFrame or a dictionary of lists).
    *   Update the main `infercnvpy.io.read()` function to accept an optional `breakpoint_file` argument.

2.  **Modify CNV Inference:**
    *   In `infercnvpy.infercnv`, modify the `infer_cnv` function to accept the breakpoint data as an optional argument.
    *   If breakpoint data is provided, the function should iterate through each segment and calculate the average expression of the genes within that segment.
    *   The CNV state for each segment will then be determined based on this average expression, relative to the reference cells.

3.  **Update Plotting Functions:**
    *   In `infercnvpy.pl`, modify the `heatmap` function to accept the breakpoint data as an optional argument.
    *   If breakpoint data is provided, the function should draw vertical lines on the heatmap to indicate the segment boundaries.

## Testing Strategy

1.  **Unit Tests:**
    *   Write unit tests for the new breakpoint file reader function.
    *   Write unit tests for the modified CNV inference function to ensure it correctly calculates segment-based CNV calls.

2.  **Integration Tests:**
    *   Create a small test dataset with known CNVs and corresponding breakpoint information.
    *   Run the entire `infercnvpy` pipeline on this dataset and verify that the inferred CNVs match the expected output.

3.  **Comparison with Existing Methods:**
    *   Compare the results of the new segment-based approach with the existing sliding window approach on a real-world dataset.
    *   This will help to demonstrate the improved accuracy and biological relevance of the new method.
