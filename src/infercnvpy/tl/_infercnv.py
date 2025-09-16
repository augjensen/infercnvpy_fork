import itertools
import re
from collections.abc import Sequence
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.sparse
from anndata import AnnData
from scanpy import logging
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from scipy.sparse import issparse

from infercnvpy._util import _ensure_array
from infercnvpy.io import read_breakpoints


def infercnv(
    adata: AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    reference: np.ndarray | None = None,
    lfc_clip: float = 3,
    window_size: int = 100,
    step: int = 10,
    dynamic_threshold: float | None = 1.5,
    exclude_chromosomes: Sequence[str] | None = ("chrX", "chrY"),
    chunksize: int = 5000,
    n_jobs: int | None = None,
    inplace: bool = True,
    layer: str | None = None,
    key_added: str = "cnv",
    calculate_gene_values: bool = False,
    breakpoint_file: str | None = None,
    min_genes_per_segment: int | None = None,
    add_segment_id_to_var: bool = False,
) -> None | tuple[dict, scipy.sparse.csr_matrix, np.ndarray | None]:
    """Infer Copy Number Variation (CNV) by averaging gene expression over genomic regions.

    This method is heavily inspired by `infercnv <https://github.com/broadinstitute/inferCNV/>`_
    but more computationally efficient. The method is described in more detail
    in on the :ref:`infercnv-method` page.

    There, you can also find instructions on how to :ref:`prepare input data <input-data>`.

    Parameters
    ----------
    adata
        annotated data matrix
    reference_key
        Column name in adata.obs that contains tumor/normal annotations.
        If this is set to None, the average of all cells is used as reference.
    reference_cat
        One or multiple values in `adata.obs[reference_key]` that annotate
        normal cells.
    reference
        Directly supply an array of average normal gene expression. Overrides
        `reference_key` and `reference_cat`.
    lfc_clip
        Clip log fold changes at this value
    window_size
        size of the running window (number of genes in to include in the window)
    step
        only compute every nth running window where n = `step`. Set to 1 to compute
        all windows.
    dynamic_threshold
        Values `< dynamic threshold * STDDEV` will be set to 0, where STDDEV is
        the stadard deviation of the smoothed gene expression. Set to `None` to disable
        this step.
    exclude_chromosomes
        List of chromosomes to exclude. The default is to exclude genosomes.
    chunksize
        Process dataset in chunks of cells. This allows to run infercnv on
        datasets with many cells, where the dense matrix would not fit into memory.
    n_jobs
        Number of jobs for parallel processing. Default: use all cores.
        Data will be submitted to workers in chunks, see `chunksize`.
    inplace
        If True, save the results in adata.obsm, otherwise return the CNV matrix.
    layer
        Layer from adata to use. If `None`, use `X`.
    key_added
        Key under which the cnv matrix will be stored in adata if `inplace=True`.
        Will store the matrix in `adata.obsm["X_{key_added}"] and additional information
        in `adata.uns[key_added]`.
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
        As many genes will be included in each segment the resultant per gene value will be an average of the genes included in the segment.
        Additionally not all genes will be included in the per gene CNV, due to the window size and step size not always being a multiple of
        the number of genes. Any genes not included in the per gene CNV will be filled with NaN.
        Note this will significantly increase the memory and computation time, it is recommended to decrease the chunksize to ~100 if this is set to True.
    breakpoint_file
        Path to a file with breakpoint data. If provided, the CNV inference will be based on the segments defined in this file.
        The file should be a tab-separated file with the following columns: `chromosome`, `start`, `end`, `segment_id`.
    min_genes_per_segment
        If `breakpoint_file` is provided, segments with fewer than this number of genes will be set to NaN.
    add_segment_id_to_var
        If True, and `breakpoint_file` is provided, add a `segment` column to `adata.var` 
        that maps each gene to its segment.


    Returns
    -------
    Depending on inplace, either return the smoothed and denoised gene expression
    matrix sorted by genomic position, or add it to adata.
    """
    if not adata.var_names.is_unique:
        raise ValueError("Ensure your var_names are unique!")
    if {"chromosome", "start", "end"} - set(adata.var.columns) != set():
        raise ValueError(
            "Genomic positions not found. There need to be `chromosome`, `start`, and `end` columns in `adata.var`. "
        )

    var_mask = adata.var["chromosome"].isnull()
    if np.sum(var_mask):
        logging.warning(f"Skipped {np.sum(var_mask)} genes because they don't have a genomic position annotated. ")  # type: ignore
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var["chromosome"].isin(exclude_chromosomes)

    tmp_adata = adata[:, ~var_mask]

    if breakpoint_file is not None:
        breakpoints = read_breakpoints(breakpoint_file)
        if "segment_id" not in breakpoints.columns:
            breakpoints["segment_id"] = (
                breakpoints["seg_chr"].astype(str)
                + ":"
                + breakpoints["seg_start"].astype(str)
                + "-"
                + breakpoints["seg_end"].astype(str)
            )
        breakpoints = breakpoints.drop_duplicates(subset=["seg_chr", "seg_start", "seg_end"])

        # Filter breakpoints to only include chromosomes present in the data
        chromosomes_in_data = [
            x for x in tmp_adata.var["chromosome"].unique() if isinstance(x, str) and x.startswith("chr") and x != "chrM"
        ]
        breakpoints = breakpoints[breakpoints["seg_chr"].isin(chromosomes_in_data)]

    else:
        breakpoints = None

    if add_segment_id_to_var and breakpoints is not None:
        adata.var["segment"] = None
        for i, segment in breakpoints.iterrows():
            genes_in_segment = (
                (adata.var["chromosome"] == segment["seg_chr"])
                & (adata.var["start"] >= segment["seg_start"])
                & (adata.var["end"] <= segment["seg_end"])
            )
            adata.var.loc[genes_in_segment, "segment"] = segment["segment_id"]

    reference = _get_reference(adata, reference_key, reference_cat, reference, layer)[:, ~var_mask]

    expr = tmp_adata.X if layer is None else tmp_adata.layers[layer]

    if scipy.sparse.issparse(expr):
        expr = expr.tocsr()

    var = tmp_adata.var.loc[:, ["chromosome", "start", "end"]]  # type: ignore

    chr_pos, chunks, convolved_dfs = zip(
        *process_map(
            _infercnv_chunk,
            [expr[i : i + chunksize, :] for i in range(0, adata.shape[0], chunksize)],
            itertools.repeat(var),
            itertools.repeat(reference),
            itertools.repeat(lfc_clip),
            itertools.repeat(window_size),
            itertools.repeat(step),
            itertools.repeat(dynamic_threshold),
            itertools.repeat(calculate_gene_values),
            itertools.repeat(breakpoints),
            itertools.repeat(min_genes_per_segment),
            tqdm_class=tqdm,
            max_workers=cpu_count() if n_jobs is None else n_jobs,
        ),
        strict=False,
    )

    res = scipy.sparse.vstack(chunks)

    chr_pos = chr_pos[0]

    if calculate_gene_values:
        per_gene_df = pd.concat(convolved_dfs, axis=0)
        # Ensure the DataFrame has the correct row index
        per_gene_df.index = adata.obs.index
        # Ensure the per gene CNV matches the adata var (genes) index, any genes
        # that are not included in the CNV will be filled with NaN
        per_gene_df = per_gene_df.reindex(columns=adata.var_names, fill_value=np.nan)
        # This needs to be a numpy array as colnames are too large to save in anndata
        per_gene_mtx = per_gene_df.values
    else:
        per_gene_mtx = None

    if inplace:
        adata.obsm[f"X_{key_added}"] = res
        adata.uns[key_added] = {"chr_pos": chr_pos}

        if calculate_gene_values:
            adata.layers[f"gene_values_{key_added}"] = per_gene_mtx

        # Replace NaN with 0 for neighbor computation
        if breakpoints is not None and min_genes_per_segment is not None:
            X = adata.obsm[f"X_{key_added}"]
            if issparse(X):
                X = X.toarray()

            # find columns that are all nan, which correspond to segments with too few genes
            nan_cols = np.all(np.isnan(X), axis=0)

            if np.any(nan_cols):
                # get the segment ids for these columns
                # The columns of the result matrix are sorted by chromosome (natural sort)
                # and then by segment start position. We need to sort the breakpoints dataframe
                # in the same way to map the column indices to segment IDs.
                sorted_chromosomes = _natural_sort(
                    [x for x in tmp_adata.var["chromosome"].unique() if x.startswith("chr") and x != "chrM"]
                )

                # The read_breakpoints function is expected to return a dataframe with
                # 'seg_chr', 'seg_start', and 'segment_id' columns.
                breakpoints["seg_chr_cat"] = pd.Categorical(
                    breakpoints["seg_chr"], categories=sorted_chromosomes, ordered=True
                )
                sorted_breakpoints = breakpoints.sort_values(["seg_chr_cat", "seg_start"]).reset_index(drop=True)

                # store the breakpoints dataframe
                adata.uns[key_added]["breakpoints"] = sorted_breakpoints.to_dict("list")

                # get the segment ids of the nan columns
                low_gene_segment_ids = sorted_breakpoints.loc[nan_cols, "segment_id"].tolist()

                # store the list of low-gene segments
                adata.uns[key_added]["low_gene_segments"] = low_gene_segment_ids

            # Replace NaN with 0 for neighbor computation
            X[np.isnan(X)] = 0
            adata.obsm[f"X_{key_added}"] = X

    else:
        return chr_pos, res, per_gene_mtx


def _natural_sort(l: Sequence):
    """Natural sort without third party libraries.

    Adapted from: https://stackoverflow.com/a/4836734/2340703
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(l, key=alphanum_key)


def _running_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 50,
    step: int = 10,
    gene_list: list = None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Compute a pyramidially weighted running mean.

    Densifies the matrix. Use `step` and `chunksize` to save memory.

    Parameters
    ----------
    x
        matrix to work on
    n
        Length of the running window
    step
        only compute running windows every `step` columns, e.g. if step is 10
        0:99, 10:109, 20:119 etc. Saves memory.
    gene_list
        List of gene names to be used in the convolution
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
    """
    if n < x.shape[1]:  # regular convolution: the filter is smaller than the #genes
        r = np.arange(1, n + 1)
        pyramid = np.minimum(r, r[::-1])
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        ## get the indices of the genes used in the convolution
        convolution_indices = get_convolution_indices(x, n)[np.arange(0, smoothed_x.shape[1], step)]
        ## Pull out the genes used in the convolution
        convolved_gene_names = gene_list[convolution_indices]
        smoothed_x = smoothed_x[:, np.arange(0, smoothed_x.shape[1], step)]

        if calculate_gene_values:
            convolved_gene_values = _calculate_gene_averages(convolved_gene_names, smoothed_x)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values

    else:  # If there is less genes than the window size, set the window size to the number of genes and perform a single convolution
        n = x.shape[1]  # set the filter size to the number of genes
        r = np.arange(1, n + 1)
        ## As we are only doing one convolution the values should be equal
        pyramid = np.array([1] * n)
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode="valid"),
            axis=1,
            arr=x,
        ) / np.sum(pyramid)

        if calculate_gene_values:
            ## As all genes are used the convolution the values are identical for all genes
            convolved_gene_values = pd.DataFrame(np.repeat(smoothed_x, len(gene_list), axis=1), columns=gene_list)
        else:
            convolved_gene_values = None

        return smoothed_x, convolved_gene_values


def _calculate_gene_averages(
    convolved_gene_names: np.ndarray,
    smoothed_x: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate the average value of each gene in the convolution

    Parameters
    ----------
    convolved_gene_names
        A numpy array with the gene names used in the convolution
    smoothed_x
        A numpy array with the smoothed gene expression values

    Returns
    -------
    convolved_gene_values
        A DataFrame with the average value of each gene in the convolution
    """
    ## create a dictionary to store the gene values per sample
    gene_to_values = {}
    # Calculate the number of genes in each convolution, will be same as the window size default=100
    length = len(convolved_gene_names[0])
    # Convert the flattened convolved gene names to a list
    flatten_list = list(convolved_gene_names.flatten())

    # For each sample in smoothed_x find the value for each gene and store it in a dictionary
    for sample, row in enumerate(smoothed_x):
        # Create sample level in the dictionary
        if sample not in gene_to_values:
            gene_to_values[sample] = {}
        # For each gene in the flattened gene list find the value and store it in the dictionary
        for i, gene in enumerate(flatten_list):
            if gene not in gene_to_values[sample]:
                gene_to_values[sample][gene] = []
            # As the gene list has been flattend we can use the floor division of the index
            # to get the correct position of the gene to get the value and store it in the dictionary
            gene_to_values[sample][gene].append(row[i // length])

    for sample in gene_to_values:
        for gene in gene_to_values[sample]:
            gene_to_values[sample][gene] = np.mean(gene_to_values[sample][gene])

    convolved_gene_values = pd.DataFrame(gene_to_values).T
    return convolved_gene_values


def get_convolution_indices(x, n):
    indices = []
    for i in range(x.shape[1] - n + 1):
        indices.append(np.arange(i, i + n))
    return np.array(indices)


def _running_mean_by_chromosome(
    expr, var, window_size, step, calculate_gene_values
) -> tuple[dict, np.ndarray, pd.DataFrame | None]:
    """Compute the running mean for each chromosome independently. Stack the resulting arrays ordered by chromosome.

    Parameters
    ----------
    expr
        A gene expression matrix, appropriately preprocessed
    var
        The var data frame of the associated AnnData object
    window_size
        size of the running window (number of genes in to include in the window)
    step
        only compute every nth running window where n = `step`. Set to 1 to compute
        all windows.

    Returns
    -------
    chr_start_pos
        A Dictionary mapping each chromosome to the index of running_mean where
        this chromosome begins.
    running_mean
        A numpy array with the smoothed gene expression, ordered by chromosome
        and genomic position
    """
    chromosomes = _natural_sort([x for x in var["chromosome"].unique() if x.startswith("chr") and x != "chrM"])

    running_means = [
        _running_mean_for_chromosome(chr, expr, var, window_size, step, calculate_gene_values) for chr in chromosomes
    ]

    running_means, convolved_dfs = zip(*running_means, strict=False)

    chr_start_pos = {}
    for chr, i in zip(chromosomes, np.cumsum([0] + [x.shape[1] for x in running_means]), strict=False):
        chr_start_pos[chr] = i

    ## Concatenate the gene dfs
    if calculate_gene_values:
        convolved_dfs = pd.concat(convolved_dfs, axis=1)

    return chr_start_pos, np.hstack(running_means), convolved_dfs


def _running_mean_for_chromosome(chr, expr, var, window_size, step, calculate_gene_values):
    genes = var.loc[var["chromosome"] == chr].sort_values("start")
    tmp_x = expr[:, var.index.get_indexer(genes.index)]
    x_conv, convolved_gene_values = _running_mean(
        tmp_x, n=window_size, step=step, gene_list=genes.index.values, calculate_gene_values=calculate_gene_values
    )

    return x_conv, convolved_gene_values


def _segment_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    segments: pd.DataFrame,
    gene_list: pd.DataFrame,
    min_genes_per_segment: int | None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """
    Compute the mean expression for each segment.

    Densifies the matrix.

    Parameters
    ----------
    x
        matrix to work on
    segments
        A DataFrame with segment information for a specific chromosome.
    gene_list
        A DataFrame with gene information (including start and end positions).
    min_genes_per_segment
        Minimum number of genes required for a segment to be considered valid.
    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers["gene_values_{key_added}"]`.
    """
    segment_means = []
    for _, segment in segments.iterrows():
        genes_in_segment = gene_list[
            (gene_list["start"] >= segment["seg_start"]) & (gene_list["end"] <= segment["seg_end"])
        ].index
        if not genes_in_segment.any() or (min_genes_per_segment is not None and len(genes_in_segment) < min_genes_per_segment):
            # Handle cases where a segment contains no genes or fewer than the minimum required
            segment_means.append(np.full((x.shape[0], 1), np.nan))
            continue
        gene_indices = [gene_list.index.get_loc(gene) for gene in genes_in_segment]
        segment_means.append(np.mean(x[:, gene_indices], axis=1, keepdims=True))

    smoothed_x = np.hstack(segment_means)

    if calculate_gene_values:
        # For simplicity, we assign the segment mean to all genes in the segment.
        # A more sophisticated approach could be implemented here.
        convolved_gene_values = pd.DataFrame(index=np.arange(x.shape[0]), columns=gene_list.index)
        for i, (_, segment) in enumerate(segments.iterrows()):
            genes_in_segment = gene_list[
                (gene_list["start"] >= segment["seg_start"]) & (gene_list["end"] <= segment["seg_end"])
            ].index
            for gene in genes_in_segment:
                convolved_gene_values[gene] = smoothed_x[:, i]
    else:
        convolved_gene_values = None

    return smoothed_x, convolved_gene_values


def _segment_mean_by_chromosome(
    expr, var, breakpoints, min_genes_per_segment, calculate_gene_values
) -> tuple[dict, np.ndarray, pd.DataFrame | None]:
    """Compute the segment mean for each chromosome independently. Stack the resulting arrays ordered by chromosome.

    Parameters
    ----------
    expr
        A gene expression matrix, appropriately preprocessed
    var
        The var data frame of the associated AnnData object
    breakpoints
        A DataFrame with breakpoint information.
    min_genes_per_segment
        Minimum number of genes required for a segment to be considered valid.
    calculate_gene_values
        If True per gene CNVs will be calculated.

    Returns
    -------
    chr_start_pos
        A Dictionary mapping each chromosome to the index of segment_mean where
        this chromosome begins.
    segment_mean
        A numpy array with the smoothed gene expression, ordered by chromosome
        and genomic position
    """
    chromosomes = _natural_sort([x for x in var["chromosome"].unique() if x.startswith("chr") and x != "chrM"])

    segment_means = []
    convolved_dfs = []
    for chr in chromosomes:
        genes = var.loc[var["chromosome"] == chr].sort_values("start")
        tmp_x = expr[:, var.index.get_indexer(genes.index)]
        
        chr_breakpoints = breakpoints[breakpoints["seg_chr"] == chr]
        
        x_conv, convolved_gene_values = _segment_mean(
            tmp_x, segments=chr_breakpoints, gene_list=genes, min_genes_per_segment=min_genes_per_segment, calculate_gene_values=calculate_gene_values
        )
        segment_means.append(x_conv)
        convolved_dfs.append(convolved_gene_values)

    chr_start_pos = {}
    for chr, i in zip(chromosomes, np.cumsum([0] + [x.shape[1] for x in segment_means]), strict=False):
        chr_start_pos[chr] = i

    ## Concatenate the gene dfs
    if calculate_gene_values:
        convolved_dfs = pd.concat(convolved_dfs, axis=1)

    return chr_start_pos, np.hstack(segment_means), convolved_dfs


def _get_reference(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: None | str | Sequence[str],
    reference: np.ndarray | None,
    layer: str | None,
) -> np.ndarray:
    """Parameter validation extraction of reference gene expression.

    If multiple reference categories are given, compute the mean per
    category.

    Returns a 2D array with reference categories in rows, cells in columns.
    If there's just one category, it's still a 2D array.
    """
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if reference is None:
        if reference_key is None or reference_cat is None:
            logging.warning(
                "Using mean of all cells as reference. For better results, "
                "provide either `reference`, or both `reference_key` and `reference_cat`. "
            )  # type: ignore
            reference = np.mean(X, axis=0)

        else:
            obs_col = adata.obs[reference_key]
            if isinstance(reference_cat, str):
                reference_cat = [reference_cat]
            reference_cat = np.array(reference_cat)
            reference_cat_in_obs = np.isin(reference_cat, obs_col)
            if not np.all(reference_cat_in_obs):
                raise ValueError(
                    "The following reference categories were not found in "
                    "adata.obs[reference_key]: "
                    f"{reference_cat[~reference_cat_in_obs]}"
                )

            reference = np.vstack([np.mean(X[obs_col.values == cat, :], axis=0) for cat in reference_cat])

    if reference.ndim == 1:
        reference = reference[np.newaxis, :]

    if reference.shape[1] != adata.shape[1]:
        raise ValueError("Reference must match the number of genes in AnnData. ")

    return reference


def _infercnv_chunk(tmp_x, var, reference, lfc_cap, window_size, step, dynamic_threshold, calculate_gene_values=False, breakpoints=None, min_genes_per_segment=None):
    """The actual infercnv work is happening here.

    Process chunks of serveral thousand genes independently since this
    leads to (temporary) densification of the matrix.

    Parameters see `infercnv`.
    """
    # Step 1 - compute log fold change. This densifies the matrix.
    # Per default, use "bounded" difference calculation, if multiple references
    # are available. This mitigates cell-type specific biases (HLA, IGHG, ...)
    if reference.shape[0] == 1:
        x_centered = tmp_x - reference[0, :]
    else:
        ref_min = np.min(reference, axis=0)
        ref_max = np.max(reference, axis=0)
        # entries that are between the two "bounds" are considered having a logFC of 0.
        x_centered = np.zeros(tmp_x.shape, dtype=tmp_x.dtype)
        above_max = tmp_x > ref_max
        below_min = tmp_x < ref_min
        x_centered[above_max] = _ensure_array(tmp_x - ref_max)[above_max]
        x_centered[below_min] = _ensure_array(tmp_x - ref_min)[below_min]

    x_centered = _ensure_array(x_centered)
    # Step 2 - clip log fold changes
    x_clipped = np.clip(x_centered, -lfc_cap, lfc_cap)
    
    # Step 3 - smooth by genomic position
    if breakpoints is not None:
        chr_pos, x_smoothed, conv_df = _segment_mean_by_chromosome(
            x_clipped, var, breakpoints=breakpoints, min_genes_per_segment=min_genes_per_segment, calculate_gene_values=calculate_gene_values
        )
    else:
        chr_pos, x_smoothed, conv_df = _running_mean_by_chromosome(
            x_clipped, var, window_size=window_size, step=step, calculate_gene_values=calculate_gene_values
        )
        
    # Step 4 - center by cell
    x_res = x_smoothed - np.nanmedian(x_smoothed, axis=1)[:, np.newaxis]
    if calculate_gene_values:
        gene_res = conv_df - np.nanmedian(conv_df, axis=1)[:, np.newaxis]
    else:
        gene_res = None

    # step 5 - standard deviation based noise filtering
    if dynamic_threshold is not None:
        noise_thres = dynamic_threshold * np.nanstd(x_res)
        x_res[np.abs(x_res) < noise_thres] = 0
        if calculate_gene_values:
            gene_res[np.abs(gene_res) < noise_thres] = 0

    x_res = scipy.sparse.csr_matrix(x_res)

    return chr_pos, x_res, gene_res
