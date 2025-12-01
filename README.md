# Traffic Forecasting & Allocation Code

This repository contains the code used for service traffic forecasting and an example resource allocation strategy.

The code is organized into:

- `Predict/`: training and testing code for the traffic predictor.
- Top-level scripts such as `strategy.py`: an example that consumes the prediction results.

---

## 1. Directory Structure

A typical layout:

```text
.
├── Predict/
│   ├── config.py
│   ├── dataset.py
│   ├── seasonality.py
│   ├── model.py
│   ├── copula.py
│   ├── train.py
│   └── test.py
├── strategy.py
└── README.md
```

> Paths in `config.py` assume that commands are run from inside `Predict/` by default.

---

## 2. Environment

- Python ≥ 3.8
- Recommended packages:
  - `torch`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `tqdm`
  - `joblib`
  - `scipy`

Example installation:

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm joblib scipy
```

(You can also use `conda` or your own environment manager.)

---

## 3. Data Format & Location

### 3.1 Files

By default (see `Predict/config.py`):

- Training data: `Predict/train_dataset.csv`
- Test data: `Predict/test_dataset.csv`

You can change these paths via `Config.train_file` and `Config.test_file`.

### 3.2 Required Columns

Both CSVs must include:

- `Timestamp`  
  - Discrete time index or sortable time indicator.  
  - Must be **monotonically increasing** after preprocessing.

- Per-service traffic columns  
  - One column per service, with names ending in `"_Arrivals"`, for example:
    - `SMS_in_Arrivals`
    - `SMS_out_Arrivals`
    - `Call_in_Arrivals`
    - `Call_out_Arrivals`
    - `Internet_Arrivals`

All service columns are automatically detected in `train.py` by selecting the columns whose names end with `"_Arrivals"`.

A minimal example:

```text
Timestamp,SMS_in_Arrivals,SMS_out_Arrivals,Call_in_Arrivals,Call_out_Arrivals,Internet_Arrivals
0,12,9,8,10,35
1,15,11,9,12,40
2,13,10,7,11,38
...
```

You can add other columns (e.g., spatial grid identifiers) if needed; they will be handled in the data loading code.

---

## 4. Configuration (`Predict/config.py`)

`Config` centralizes basic settings for data, model and training. Important fields include:

- **Data & paths**
  - `train_file` / `test_file`: input CSV paths.
  - `results_dir`: directory for all outputs (default: `results/`).
  - `model_save_path`: model checkpoint path (e.g., `results/traffic_model.pth`).
  - `scaler_save_path`: normalization parameters file.
  - `seasonality_save_path`: parameters for the seasonality module.
  - `copula_save_path`: parameters for the dependence module.

- **Model / training**
  - `seq_len`, `pred_len`: input history length and prediction horizon.
  - `batch_size`, `num_epochs`, `learning_rate`, etc.
  - `quantiles`: list of quantile levels.
  - `device`: `"cuda"` or `"cpu"`.

- **Data-related**
  - `period`: time-series period used by `seasonality.py` (e.g., number of steps per day).
  - `service_cols`: automatically populated in `train.py` based on `*_Arrivals` columns.

You usually only need to edit paths, sequence lengths, period, and device settings to match your environment.

---

## 5. Training (`Predict/train.py`)

From the repository root:

```bash
cd Predict
python train.py
```

This script:

1. Loads the training CSV specified by `Config.train_file`.
2. Detects all service columns whose names end with `"_Arrivals"` and aggregates traffic by `Timestamp`.
3. Uses `seasonality.py`:
   - `SeasonalityAnalyzer` and related utilities are called inside `train.py` to build time/seasonality-related features.
4. Normalizes the data and constructs datasets using `dataset.py`.
5. Builds and trains the forecasting model defined in `model.py`.
6. Uses `copula.py`:
   - `ResidualDependenceModeler` is called inside `train.py` to fit a dependence model on residuals and save its parameters.
7. Saves:
   - The trained model checkpoint (`Config.model_save_path`).
   - Normalization parameters (`Config.scaler_save_path`).
   - Seasonality parameters (`Config.seasonality_save_path`).
   - Dependence model parameters (`Config.copula_save_path`).
   - Training logs/plots under `Config.results_dir`.

---

## 6. Testing & Prediction Outputs (`Predict/test.py`)

From the repository root:

```bash
cd Predict
python test.py
```

`test.py` will:

1. Load:
   - The model and scaler using paths in `config.py`.
   - Seasonality parameters (produced by `seasonality.py` during training).
   - Dependence model parameters (produced by `copula.py` during training).
2. Load both training and test CSVs, aggregate them by `Timestamp`, and prepare the time-series inputs and features.
3. Run rolling predictions on the test horizon with the trained model.
4. Save results into `Config.results_dir`, typically including:
   - `quantile_predictions.csv`  
     Time-series predictions for each service and each configured quantile level.
   - `quantile_metrics.csv`  
     Basic evaluation metrics for the predictions.
   - `joint_demand_quantiles.csv`  
     Additional time-series statistics based on the dependence module (if available).

Exact file names and paths may vary slightly depending on the current `Config` settings.

---

## 7. Example Strategy Script (`strategy.py`)

At the repository root:

```bash
python strategy.py
```

This script:

- Reads prediction-related CSV files generated by `Predict/test.py`, typically from `Predict/`’s `results_dir` (for example, `quantile_predictions.csv`, `joint_demand_quantiles.csv`).
- Uses service and slice definitions from `data.py` (not shown here; you should adapt it to your own system, total capacities, and service set).
- Produces a CSV file (for example, `resource_allocation_strategy.csv`) containing a time series of per-slice allocation decisions.

Notes:

- `strategy.py` is an example that shows how to consume the prediction outputs for allocation.
- Numerical parameters and thresholds in `strategy.py` are **scenario-dependent** and should be tuned according to your own resource scale, traffic patterns, and requirements.

---

## 8. File Overview

- `Predict/config.py`  
  Global configuration for data paths, model/training settings, and output locations.

- `Predict/dataset.py`  
  Dataset and data-loading utilities used by `train.py` and `test.py`.

- `Predict/seasonality.py`  
  Seasonality and time-feature utilities. Used in both `train.py` and `test.py` through `SeasonalityAnalyzer` and related helpers.

- `Predict/model.py`  
  Neural network model for traffic forecasting, plus associated loss and helper functions.

- `Predict/copula.py`  
  Dependence modeling and sampling utilities on residuals. Used in `train.py` to fit the model, and in `test.py` to load parameters and generate additional statistics.

- `Predict/train.py`  
  End-to-end training script: data loading, feature construction (including seasonality), model training, and dependence model fitting.

- `Predict/test.py`  
  End-to-end testing / inference script: load all saved components, run rolling predictions on the test horizon, and write prediction/metric/statistics CSVs.

- `strategy.py`  
  Example script that reads the prediction outputs and generates one feasible resource allocation plan as a time series. Parameters inside are illustrative and should be adjusted for your own environment.
