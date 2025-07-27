from datasets import load_dataset

def load_ctrate_reports():
    """Load CT-RATE reports split from Hugging Face."""
    return load_dataset("ibrahimhamamci/CT-RATE", "reports")

def load_ctrate_metadata():
    """Load CT-RATE metadata split from Hugging Face."""
    return load_dataset("ibrahimhamamci/CT-RATE", "metadata")

def load_ctrate_labels():
    """Load CT-RATE labels split from Hugging Face."""
    return load_dataset("ibrahimhamamci/CT-RATE", "labels")
