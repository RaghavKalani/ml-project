# 1. Project Requirements

Before running the project, make sure you have:

- Python 3.10 or newer
- Visual Studio Code
- VS Code extensions:
  - Python
  - Jupyter
- The libraries listed in `requirements.txt`
  - This includes the report-generation libraries `python-docx` and `reportlab`

# 2. Create Virtual Environment

## Windows
Open a terminal in the project folder and run:

```bash
python -m venv .venv
```

## Mac/Linux
Open a terminal in the project folder and run:

```bash
python3 -m venv .venv
```

# 3. Activate Virtual Environment

## Windows

```bash
.\.venv\Scripts\activate
```

## Mac/Linux

```bash
source .venv/bin/activate
```

# 4. Install Dependencies

After activating the virtual environment, install the required packages:

```bash
pip install -r requirements.txt
```

# 5. Verify Installation

Run the following checks:

```bash
python --version
pip --version
python -c "import pandas, numpy, matplotlib, seaborn, scipy, statsmodels, sklearn; print('All imports successful')"
python -c "import docx, reportlab; print('Report libraries successful')"
```

If the command prints `All imports successful`, your environment is ready.

# 6. Running the Notebook

1. Open VS Code.
2. Open this project folder.
3. Open `notebooks/01_EDA.ipynb`.
4. Select the Python interpreter from `.venv` if prompted.
5. In the notebook toolbar, choose the correct kernel.
6. Run each cell from top to bottom.

If you want to open Jupyter Notebook directly in VS Code, click the notebook file and run the cells there.

# 7. Folder Structure

- `Stroke prediction dataset/` stores the original CSV dataset.
- `notebooks/` stores the EDA notebook.
- `reports/` stores the written markdown report.
- `figures/` stores all saved charts and plots.
- `src/` stores reusable Python code for loading data, creating plots, and generating the report.
- `requirements.txt` lists the Python dependencies.
- `README.md` summarizes the project.
- `START.md` is this beginner guide.

# 8. Common Errors

## ModuleNotFoundError
This means a package is missing. Activate the virtual environment again and run:

```bash
pip install -r requirements.txt
```

## Wrong Python Interpreter
If VS Code uses the wrong Python version, open the Command Palette and select `Python: Select Interpreter`. Choose the interpreter inside `.venv`.

## Missing Dataset
If the notebook cannot find the CSV file, confirm that this path exists:

`Stroke prediction dataset/healthcare-dataset-stroke-data.csv`

## Jupyter Kernel Not Found
If the notebook cannot start a kernel, reinstall the kernel package:

```bash
pip install ipykernel
```

Then restart VS Code and select the correct kernel again.

## Permission Errors
If a file cannot be written, make sure no other program is locking the file and that you have permission to write in the project folder.

# 9. Output

- All generated figures are saved in `figures/`
- The analysis report is saved in `reports/EDA_Report.md`

# 10. Expected Results

After running the project successfully, you should see:

- A completed notebook with markdown explanations and executed code cells
- Saved figures for target analysis, numerical analysis, categorical analysis, correlation, outliers, and distribution checks
- A professional EDA report in markdown format
- A clear project structure that can be reused for the modeling phase
