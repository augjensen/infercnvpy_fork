import pandas as pd

def read_breakpoints(file_path: str) -> pd.DataFrame:
    """
    Read a gene segment overlap file.

    The file should be a comma-separated file with the following columns:
    `SYMBOL`, `gene_chr`, `gene_start`, `gene_end`, `seg_chr`, `seg_start`, `seg_end`, `seg_arm`

    Parameters
    ----------
    file_path
        Path to the gene segment overlap file.

    Returns
    -------
    A pandas DataFrame with the gene segment overlap information.
    """
    return pd.read_csv(file_path)
