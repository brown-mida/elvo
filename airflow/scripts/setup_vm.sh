# You should run this only once, when first setting up airflow
# on a new VM. The VM should have Python 3.6 installed.
#!/usr/bin/env bash

export AIRFLOW_HOME=$PWD
echo export AIRFLOW_HOME=$PWD >> ~/.bashrc # So $AIRFLOW_HOME is permanently set

python3 -m venv venv
source $PWD/venv/bin/activate

pip install -r requirements.txt

airflow initdb
