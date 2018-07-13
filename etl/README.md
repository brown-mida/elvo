Contains [ETL](http://datawarehouse4u.info/ETL-process.html) code. This
does not include the ad-hoc preprocessing done just before training a
ML model.

For now we'll be using environment variables in place of paths,
as it works the best with Airflow. We can change this in the future.

To develop locally, install the base requirements
and also run `pip install apache-airflow[slack,gcp_api]==1.8.2`

Unless you plan to do lots of data pipeline work,
it is recommended that you use SFTP/SCP to test your workflows on
Airflow directly instead of setting up Airflow locally.

See the `docs/AIRFLOW_GUIDE.md` guide for more info on using Airflow.