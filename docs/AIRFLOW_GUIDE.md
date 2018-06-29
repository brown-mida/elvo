## Step 1: Getting the Script to airflow

Make a PR and push your changes to the airflow branch.

After your changes are on `origin/airflow`, run the following commmands in your terminal:

```
gcloud compute ssh lukezhu@airflow
cd elvo-analysis
git pull
```

Your script should now be in the etl folder.

## Step 2: Creating a DAG

Before this step familiarize yourself with the
[Airflow tutorial](https://airflow.apache.org/tutorial.html).

Create file which contains the DAG that runs your script. The
easiest way to test your DAG is through the web UI (by turning on
the dag, clicking trigger DAG and then checking the task instance logs).

See http://michal.karzynski.pl/blog/2017/03/19/developing-workflows-with-apache-airflow/
and https://docs.astronomer.io/v2/apache_airflow/best-practices-guide.html
for more tips on how to get started.

For this part, just push your commits to `airflow`.


## Tips:
- Prefer `PythonOperator` over `bash operator`
- Prefer absolute paths over relative paths.


## Notes

- Environment variables are set in /home/lukezhu/.bashrc
- Use the airflow.cfg in /home/lukezhu/elvo-analysis
- We should periodically rebase this branch onto master