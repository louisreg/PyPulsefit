import pytest
import pandas as pd
from matplotlib import pyplot as plt

from pypulsefit.patch_clamp import Dataset, Sweep, RecordingType

# --- Mock .asc content with two sweeps ---
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
    file_path = tmp_path / "mock.asc"
    file_path.write_text(MOCK_ASC)
    return str(file_path)


def test_dataset_loading(tmp_asc_file):
    """Test that Dataset.from_asc loads sweeps correctly."""
    dataset = Dataset.from_asc(tmp_asc_file, recording_type=RecordingType.INTRACELLULAR)
    
    # Check type
    assert isinstance(dataset, Dataset)
    assert isinstance(dataset.sweeps, dict)
    assert len(dataset.sweeps) == 2
    
    # Check sweep keys
    assert "Sweep 1_1_1" in dataset.sweeps
    assert "Sweep 1_2_1" in dataset.sweeps
    
    # Check Sweep objects
    sweep1 = dataset["Sweep 1_1_1"]
    assert isinstance(sweep1, Sweep)
    assert isinstance(sweep1.time, pd.Series)
    assert isinstance(sweep1.voltage, pd.Series)
    assert isinstance(sweep1.current, pd.Series)
    assert len(sweep1.time) == 5
    
    sweep2 = dataset["Sweep 1_2_1"]
    assert len(sweep2.time) == 4


def test_sweep_plot(tmp_asc_file):
    """Ensure that the Sweep.plot method runs without errors."""
    dataset = Dataset.from_asc(tmp_asc_file, recording_type=RecordingType.INTRACELLULAR)
    sweep = dataset["Sweep 1_1_1"]
    
    fig, ax = plt.subplots()
    sweep.plot(ax=ax)
    
    plt.close(fig)


def test_list_sweeps(tmp_asc_file):
    """Test that Dataset.available_sweeps() returns correct sweep names."""
    dataset = Dataset.from_asc(tmp_asc_file, recording_type=RecordingType.INTRACELLULAR)
    
    sweep_names = dataset.list_sweeps()
    
    # Check type
    assert isinstance(sweep_names, list)
    
    # Check that all expected sweeps are listed
    expected_sweeps = ["Sweep 1_1_1", "Sweep 1_2_1"]
    assert set(sweep_names) == set(expected_sweeps)
    
    # Optional: check that each sweep can be accessed via __getitem__
    for name in sweep_names:
        sweep = dataset[name]
        assert sweep.name == name
        assert isinstance(sweep, Sweep)
        

def test_get_sweep_by_index(tmp_asc_file):
    """Test that Dataset.get_sweep_by_index() retrieves the correct sweep."""
    dataset = Dataset.from_asc(tmp_asc_file, recording_type=RecordingType.INTRACELLULAR)
    
    sweep0 = dataset.get_sweep_by_index(0)
    sweep1 = dataset.get_sweep_by_index(1)
    
    expected_names = ["Sweep 1_1_1", "Sweep 1_2_1"]
    
    assert sweep0.name == expected_names[0]
    assert sweep1.name == expected_names[1]
    
    # Test out-of-range raises IndexError
    with pytest.raises(IndexError):
        dataset.get_sweep_by_index(2)