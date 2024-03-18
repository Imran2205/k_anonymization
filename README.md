
---

# Anonymizer Script

Analysis of the Adult dataset is available in the `notebooks/k_anonymization.ipynb` file.

The `anonymizer.py` script applies the Mondrian k-anonymization algorithm to achieve k-anonymity in datasets, such as the Adult dataset, ensuring privacy protection and preventing individual re-identification. This script is designed to handle both numerical and categorical data, making it versatile for various anonymization needs.

## Getting Started

### Prerequisites

Ensure you have Python 3 installed on your system. You can check your Python version by running:

```bash
python3 --version
```

### Usage

The `anonymizer.py` script takes several arguments to specify the level of anonymity (k-value), the quasi-identifiers, the sensitive attribute, and the output file path. You can learn about all available arguments and the script's functionality by using the `--help` option:

```bash
python3 anonymizer.py --help
```

### Arguments

- `-k`, `--k_value`: The k-value for k-anonymity, determining the level of privacy.
- `-q`, `--quasi_identifiers`: A list of column names to be treated as quasi-identifiers, separated by commas.
- `-s`, `--sensitive_attribute`: The name of the sensitive attribute in the dataset.
- `-o`, `--output`: Path to save the anonymized dataset.

### Running Example

To anonymize a dataset with a k-value of 3, treating 'age' and 'hours-per-week' as quasi-identifiers, 'income' as a sensitive attribute, and specifying input and output files, you would run:

```bash
python3 anonymizer.py --k 3 --quasi_identifiers age,hours-per-week --sensitive_attribute income --output path/to/anonymized_dataset.csv
```

Replace `path/to/anonymized_dataset.csv` with the desired path for the anonymized output.

---
