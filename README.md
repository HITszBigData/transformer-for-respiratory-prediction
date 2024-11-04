# Transformer for Rispiratory Diseases Prediction

We will fine-tuning BERT to predict respiratory diseases

To get started, you need to:
1. Clone the repository:
```sh
git clone https://github.com/HITszBigData/transformer-for-respiratory-prediction.git
cd transformer-for-respiratory-prediction
```
2. Install the dependency
```sh
conda create -n bert-classify python=3.8
conda activate bert-classify
pip install -r requirements.txt
```
3. Fetch the dataset ICBHI:
```sh
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip --no-check-certificate
unzip ICBHI_final_database.zip
```