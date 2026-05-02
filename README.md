# Waveform Test V2

This project runs `waveform_test_v2.py` on the Waveform dataset using two QUBO-based feature-selection methods:

- `Muecke_QBSolv`
- `TaylorS_QBSolv`

It then evaluates the selected features with 10-fold cross-validation using logistic regression and saves the summary metrics to a CSV file.

## Files

- `waveform_test_v2.py`: main script
- `waveform.data`: standard waveform dataset
- `waveform-+noise.data`: waveform dataset with noise features
- `requirements.txt`: Python dependencies

## What the script does

When you run the script, it will:

1. Load one of the included waveform datasets.
2. Select `k=5` features using:
   - Muecke QFS with `QBSolv`
   - Taylor-style QFS with `QBSolv`
3. Evaluate the selected features with 10-fold stratified cross-validation.
4. Print the selected features, timing, and accuracy summary.
5. Save the results to a CSV file in the same folder.

## Tested environment

The script was verified in this environment with:

- Python `3.10.11`
- `numpy==1.26.4`
- `pandas==2.1.4`
- `scikit-learn==1.4.1.post1`
- `dwave-qbsolv==0.3.4`

If you have trouble installing `dwave-qbsolv` on a newer Python release, use Python 3.10 first because that version is confirmed here.

## Setup from scratch

### 1. Open a terminal in this folder

The script expects the dataset files to be in the same directory as `waveform_test_v2.py`.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## How to run

Run the script with:

```powershell
python waveform_test_v2.py
```

## Choosing the dataset variant

By default, the script uses the standard dataset:

```python
DATA_VARIANT = "standard"
```

To run the noisy version instead, open `waveform_test_v2.py` and change it to:

```python
DATA_VARIANT = "noise"
```

Available options:

- `"standard"` uses `waveform.data`
- `"noise"` uses `waveform-+noise.data`

## Output

The script prints:

- dataset shape and class balance
- selected feature indices for each feature-selection method
- 10-fold CV accuracy mean and standard deviation
- runtime per feature-selection method

It also writes a CSV file in the same folder:

- `results_waveform_standard.csv` when `DATA_VARIANT = "standard"`
- `results_waveform_noise.csv` when `DATA_VARIANT = "noise"`

If the target CSV is already open in another program, the script will automatically save to a timestamped fallback file instead of stopping with an error.

## Notes

- The script now resolves dataset and output paths relative to its own folder, so you do not need to edit an absolute path before running it.
- The console output was kept ASCII-only for better Windows terminal compatibility.
- `QBSolv` is used through the `dwave_qbsolv` Python package.
