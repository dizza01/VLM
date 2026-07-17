# Notebook policy

Notebooks are for:

- exploratory inspection;
- development-only examples;
- visual quality control;
- regenerating paper tables and figures from immutable run artifacts.

Long-running data preparation, training, inference, attribution, perturbation and scoring belong
in importable modules and command-line stages. A notebook must not be the only place an
experiment can be reproduced.

Recommended notebooks:

- `01_data_audit.ipynb`
- `02_pilot_inspection.ipynb`
- `03_results_report.ipynb`

Keep outputs small or strip them before committing. Never embed credentials or model weights.
The existing Study 1 notebook remains in the repository root during this additive migration.

