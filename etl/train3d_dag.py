import datetime

from airflow import DAG
from airflow.contrib.hooks.gcp_mlengine_hook import MLEngineHook
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from googleapiclient import errors

default_args = {
    'owner': 'luke',
    'email': 'luke_zhu@brown.edu',
    'start_date': datetime.datetime(2018, 7, 31),
}

dag = DAG(dag_id='train3d_web',
          default_args=default_args,
          description='Train a C3D model on cloud ML',
          catchup=False,
          schedule_interval=None)


# Copied from mlengine_operator
class MyMLEngineTrainingOperator(BaseOperator):
    @apply_defaults
    def __init__(self,
                 project_id,
                 job_id,
                 package_uris,
                 training_python_module,
                 training_args,
                 region,
                 gcp_conn_id='google_cloud_default',
                 delegate_to=None,
                 mode='PRODUCTION',
                 *args,
                 **kwargs):
        super(MyMLEngineTrainingOperator, self).__init__(*args, **kwargs)
        self._project_id = project_id
        self._job_id = job_id
        self._package_uris = package_uris
        self._training_python_module = training_python_module
        self._training_args = training_args
        self._region = region
        self._gcp_conn_id = gcp_conn_id
        self._delegate_to = delegate_to
        self._mode = mode

    def execute(self, context):
        job_id = self._job_id
        training_request = {
            'jobId': job_id,
            'trainingInput': {
                # TODO(luke): Added by me
                'scaleTier': 'BASIC_GPU',
                'masterType': '',  # TODO: Set the masterType
                'packageUris': self._package_uris,
                'pythonModule': self._training_python_module,
                'region': self._region,
                'args': self._training_args,
                # TODO(luke): Added by me
                'runtimeVersion': '1.4',
                'pythonVersion': '3.5',
            }
        }

        if self._mode == 'DRY_RUN':
            self.log.info('In dry_run mode.')
            self.log.info('MLEngine Training job request is: {}'.format(
                training_request))
            return

        hook = MLEngineHook(
            gcp_conn_id=self._gcp_conn_id, delegate_to=self._delegate_to)

        # Helper method to check if the existing job's training input is the
        # same as the request we get here.
        def check_existing_job(existing_job):
            return existing_job.get('trainingInput', None) == \
                   training_request['trainingInput']

        try:
            finished_training_job = hook.create_job(
                self._project_id, training_request, check_existing_job)
        except errors.HttpError:
            raise

        if finished_training_job['state'] != 'SUCCEEDED':
            self.log.error('MLEngine training job failed: {}'.format(
                str(finished_training_job)))
            raise RuntimeError(finished_training_job['errorMessage'])


# TODO: This should be a variable
package_uris = ['gs://elvos/cloud-ml/cloud3d/dist/cloudml-c3d-0.0.2.tar.gz']
job_id = 'c3d_test24'

training_op = MyMLEngineTrainingOperator(
    project_id='elvo-198322',
    job_id=job_id,
    package_uris=package_uris,
    training_python_module='trainer.task',
    training_args='',
    region='us-east1',
    task_id='train_model',
    dag=dag,
    mode='PRODUCTION'
)
