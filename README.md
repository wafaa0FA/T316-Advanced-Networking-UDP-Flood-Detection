# UDP Flood Attack Detection using Machine Learning (T316 Project)

This repository contains the implementation of my T316 Advanced Networking project.  
The goal is to detect **UDP flood attacks** in a Mininet-emulated network using Machine Learning.

The project covers:

- Traffic generation in Mininet  
- Capturing and processing PCAP files  
- Converting PCAP to CSV  
- Training ML models on the extracted features  
- Evaluating model performance  
- Running inference on new traffic traces  

---

## Repository Structure

- `infer.py`  
  Loads a trained model and runs **inference** on new CSV traffic files (classifies traffic as **normal** or **attack**).

- `train_and_evaluate.py`  
  Main script to **train** and **evaluate** machine learning models on the dataset (e.g., Random Forest, KNN).  
  Prints metrics such as accuracy, precision, recall, F1-score, and possibly saves trained models.

- `pcap_analysis.py`  
  Processes PCAP/CSV files, performs feature extraction/engineering, and prepares a labeled dataset for training.

- `pcap_to_csv.sh`  
  Shell script that converts **PCAP** files into **CSV** format using `tshark` or a similar tool.

- `mininet_scripts.py`  
  Script that sets up the **Mininet topology**, generates normal and attack traffic (UDP flood), and captures PCAP traces.

- `requirements.txt`  
  List of Python dependencies required to run the project.

> Note: Large datasets, raw PCAP files, and trained model binaries may be excluded from the repository to keep it lightweight.  
> The provided scripts implement the full processing and training pipeline.

---

## 1. Prerequisites

### System

- Linux-based environment (recommended for Mininet)
- Python 3.8+  
- Mininet installed (if you want to regenerate traffic and PCAPs)

### Python packages

Create and activate a virtual environment (optional but recommended):

bash
python3 -m venv venv
source venv/bin/activate
Install the Python dependencies:

bash
pip install -r requirements.txt
2. Generating Traffic and PCAP Files (Mininet)
If you want to regenerate the dataset from scratch using Mininet:

bash
sudo python3 mininet_scripts.py
This script should:

Build the Mininet topology (hosts and switch)

Generate normal traffic and UDP flood attack traffic (e.g., using iperf3 and ping)

Capture packets to PCAP files (for example, stored in a pcaps/ directory)

3. Converting PCAP to CSV
After generating PCAP files, convert them into CSV format:

bash
bash pcap_to_csv.sh
This script typically:

Iterates over PCAP files

Uses tshark (or similar tools) to extract fields

Saves the output as CSV files (for example, into a pcaps_csv/ directory)

Make sure tshark is installed:

bash
sudo apt-get install tshark
4. Feature Extraction and Dataset Preparation
Use pcap_analysis.py to further process the converted CSV files and build a training dataset:

bash
python3 pcap_analysis.py
This script is responsible for:

Reading CSV files generated from PCAPs

Cleaning and transforming the data (feature engineering, handling missing values, etc.)

Creating a labeled dataset, for example:

0 → normal traffic

1 → UDP flood attack

The final dataset will then be used by the training script.

5. Training and Evaluating Models
Once the dataset is prepared, run the training and evaluation script:

bash
python3 train_and_evaluate.py
This script should:

Load the prepared dataset (from CSV)

Split it into training and testing sets

Train one or more models (e.g., Random Forest, KNN)

Evaluate them using common metrics:

Accuracy

Precision

Recall

F1-score

ROC-AUC (if implemented)

Optionally save the trained model(s) (e.g., into a models/ directory)

Optionally save plots or metrics into a results/ directory

Check the console output and any generated files (e.g., confusion matrices, ROC curves, metrics summaries).

6. Running Inference on New Traffic
After training, you can classify new traffic traces using infer.py:

bash 
python3 infer.py --input <path_to_csv_file>
Example:


bash
python3 infer.py --input data/new_capture.csv
Typical responsibilities of infer.py:

Load the trained model (for example from models/)

Load and preprocess the input CSV file

Predict whether each record (or the traffic as a whole) is normal or attack

Print the classification results and/or save them to a file

If your infer.py uses different argument names (e.g. --model, --output), adjust the command accordingly.

7. Reproducibility
The pipeline can be summarized as:

Generate traffic and PCAPs using Mininet (mininet_scripts.py)

Convert PCAP → CSV (pcap_to_csv.sh)

Analyze and prepare the dataset (pcap_analysis.py)

Train and evaluate ML models (train_and_evaluate.py)

Run inference on new traces (infer.py)

Most parameters (topology, traffic rate, model hyperparameters, etc.) can be modified directly inside the corresponding scripts.

8. Project Context
This repository is part of my T316 – Advanced Networking course project:

“Machine Learning-based UDP Flood Attack Detection in a Mininet-Emulated Network.”

It demonstrates how combining network simulation with machine learning can be used to detect denial-of-service style attacks in modern networks.
