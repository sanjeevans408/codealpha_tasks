# Credit Scoring — Task1

Brief credit scoring demo that generates a synthetic dataset and trains three models (Logistic Regression, Decision Tree, Random Forest), then saves a performance report image.

**Files**
- [Creditscoring.py](Creditscoring.py)
- [requirements.txt](requirements.txt)

**Prerequisites**
- Python 3.9+ (recommended)
- A virtual environment (recommended)

**Setup (Windows)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # or use Activate.bat for cmd
pip install -r requirements.txt
```

**Setup (Unix/macOS)**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run**
```powershell
.\.venv\Scripts\python Creditscoring.py
```

The script will create an `outputs/` folder and save the report as `outputs/credit_scoring_report.png`.

**Notes**
- If you already have a working environment, installing the packages from `requirements.txt` is sufficient.
- Adjust package versions in `requirements.txt` if you need strict pinning for reproducibility.
