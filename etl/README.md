Contains [ETL](http://datawarehouse4u.info/ETL-process.html) code. This
does not include the ad-hoc preprocessing done just before training a
ML model.

For now we'll be using environment variables in place of paths,
as it works the best with Airflow. We can change this in the future.

To set up a new Airflow instance, install the base requirements
and also run `pip install apache-airflow[slack,gcp_api]==1.8.2`