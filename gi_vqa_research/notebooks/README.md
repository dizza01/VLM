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

- `00_colab_t4_backend_contract.ipynb` — executable one-item CUDA compatibility
  gate;
- `01_data_audit.ipynb`
- `02_pilot_inspection.ipynb`
- `03_results_report.ipynb`

Keep outputs small or strip them before committing. Never embed credentials or model weights.
The existing Study 1 notebook remains in the repository root during this additive migration.

## Run the Colab T4 contract

Open `00_colab_t4_backend_contract.ipynb` from the pushed GitHub repository in
Colab. Select a T4 GPU, expose a read-capable Colab secret named `HF_TOKEN`,
paste the full 40-character SHA of the commit containing the notebook and
backend, and run all cells.

The notebook does not contain the scientific implementation. It installs the
package from the same exact commit and invokes:

```bash
python -m gi_vqa.contract
```

The runner always writes `contract_report.json` after it starts, and the
notebook bundles that report, attribution arrays when produced, the diagnostic
image, environment record and `pip freeze`. Preserve the downloaded bundle
with the tested Git commit. Bootstrap and authentication failures are also
converted into a small failure report and evidence bundle before the final
notebook cell raises.

A PASS establishes only that the shared backend works in the pinned reference
environment. The fixed item is excluded from research results and reserved
from future split manifests. Do not proceed to the 20-item smoke until the
grouped split manifest has been built and audited.
