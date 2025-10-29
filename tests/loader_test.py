import pandas as pd
import pytest
from pypulsefit.loader import to_dfs

# --- Mock .asc content with two datasets ---
MOCK_ASC = """Sweep 1_1_1
5 points
"time[s]",   "trace1 [A]",   "trace2 [V]"
0.00000E+000  -2.18750E-011   5.96563E-002
4.44000E-004  -2.50000E-011   5.97812E-002
8.88000E-004  -2.81250E-011   5.97188E-002
1.33200E-003  -2.50000E-011   5.98125E-002
1.77600E-003  -2.50000E-011   5.97812E-002

Sweep 1_2_1
4 points
"time[s]",   "trace1 [A]",   "trace2 [V]"
0.00000E+000  -3.12500E-011   5.31250E-002
3.46000E-004  -2.81250E-011   5.31563E-002
6.92000E-004  -2.81250E-011   5.30938E-002
1.03800E-003  -2.81250E-011   5.31875E-002
"""

@pytest.fixture
def tmp_asc_file(tmp_path):
    """Create a temporary .asc file for testing."""
    file_path = tmp_path / "test_file.asc"
    file_path.write_text(MOCK_ASC)
    return str(file_path)


def test_to_dfs_returns_dict(tmp_asc_file):
    """Ensure the function returns a dictionary of DataFrames."""
    datasets = to_dfs(tmp_asc_file)

    assert isinstance(datasets, dict), "Expected a dictionary output"
    assert len(datasets) == 2, "Should detect 2 datasets"
    assert all(isinstance(df, pd.DataFrame) for df in datasets.values())


def test_dataset_keys(tmp_asc_file):
    """Check that the correct sweep names are detected."""
    datasets = to_dfs(tmp_asc_file)
    expected_keys = {"Sweep 1_1_1", "Sweep 1_2_1"}
    assert set(datasets.keys()) == expected_keys


def test_dataframe_structure(tmp_asc_file):
    """Ensure DataFrames have expected columns and numeric data."""
    datasets = to_dfs(tmp_asc_file)

    for name, df in datasets.items():
        assert list(df.columns) == ["time[s]", "trace1 [A]", "trace2 [V]"]
        assert not df.empty
        assert df.map(lambda x: isinstance(x, (int, float))).all().all(), f"Non-numeric data in {name}"


def test_dataframe_values(tmp_asc_file):
    """Check numeric correctness of one example value."""
    datasets = to_dfs(tmp_asc_file)
    df = datasets["Sweep 1_1_1"]
    assert pytest.approx(df.iloc[0]["trace2 [V]"], rel=1e-6) == 5.96563E-002
