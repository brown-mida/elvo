# Apache Airflow Guide
=======================

Below are instructions for getting a data pipeline deployed on our
Apache Airflow cluster.

Some advantages of Airflow are listed here:
 - you can schedule your code to run daily, weekly, etc.
 - you can more easily understand why your code is failing
 - you can use Airflow methods to send email or Slack alerts when
 your code fails

## Step 1: Scripts Review
Make sure your scripts works locally. This means writing tests for
your script.

Once you are ready, create a pull request and ask Luke to review.

## Step 2: Scripts to Dag

Install Apache Airflow and related dependencies on your dev machine (say gpu1708)
using the command `pip install .[etl]`. Familiarize yourself with the
[Airflow tutorial](https://airflow.apache.org/tutorial.html).

Now you should do the following:
1. Setup airflow on your dev machine using `airflow initdb` and other commands listed
in the tutorial. 
2. Create file which contains the DAG that runs your scripts. See `etlv2/elvo_dag` for an example.
3. Test your DAG using the Airflow CLI commands.

For more tips on how to get started, see http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow/
and https://docs.astronomer.io/v2/apache_airflow/best-practices-guide.html

Once you are ready, create a pull request and ask Luke to review.

## Tips:
- Prefer `PythonOperator` over `BashOperator`
- Prefer absolute paths over relative paths.


## Notes

- The required steps to setup a GCS VM to run Airflow from scratch are:
    - Install Python 3.6 (preferably with miniconda)
    - Git clone elvo-analysis and run `pip install -e .[cpu,etl]`
    - Set the environment variables in .bashrc and get the corresponding files to put in `secrets`
    - Copy over the `airflow.cfg` file and change the necessary params