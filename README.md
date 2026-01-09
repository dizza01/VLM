# VLM

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Environment Setup

1. **Clone/Navigate to the project directory:**
   ```bash
   cd /path/to/VLM
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   
   **On macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
   
   **On Windows:**
   ```bash
   venv\Scripts\activate
   ```

4. **Upgrade pip (recommended):**
   ```bash
   pip install --upgrade pip
   ```

5. **Install project dependencies** (when requirements.txt is available):
   ```bash
   pip install -r requirements.txt
   ```

6. **Deactivate the virtual environment** (when you're done working):
   ```bash
   deactivate
   ```

### Getting Started

Once your environment is set up, you can start working with the Jupyter notebook:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter notebook
jupyter notebook
```

Then open `Task_1_Sample_Notebook.ipynb` to get started.

### Notes

- Always activate the virtual environment before working on the project
- The `venv/` folder is included in `.gitignore` and should not be committed to version control
- If you add new dependencies, remember to update the requirements.txt file:
  ```bash
  pip freeze > requirements.txt
  ```