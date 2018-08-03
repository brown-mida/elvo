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
class CustomCloudMLTrainingOperator(BaseOperator):
    @apply_defaults
    def __init__(self,
                 project_id,
                 gcp_conn_id='google_cloud_default',
                 delegate_to=None,
                 mode='PRODUCTION',
                 *args,
                 **kwargs):
        super(CustomCloudMLTrainingOperator, self).__init__(*args, **kwargs)
        self._project_id = project_id
        self._gcp_conn_id = gcp_conn_id
        self._delegate_to = delegate_to
        self._mode = mode

    def execute(self, context):
        conf = context['dag_run'].conf
        job_id = conf['jobName']
        print('jobID: {}'.format(job_id))
        training_request = {
            'jobId': job_id,
            'trainingInput': {
                # TODO(luke): Added by me
                'scaleTier': 'BASIC_GPU',
                'packageUris': [
                    'gs://elvos/cloud-ml/cloud3d/dist/cloudml-c3d-0.0.2.tar'
                    '.gz'],
                'pythonModule': 'trainer.task',
                'region': 'us-east1',
                'args': '',
                # TODO(luke): Added by me
                'runtimeVersion': '1.4',
                'pythonVersion': '3.5',
            }
        }

        if self._mode == 'DRY_RUN':
            self.info('In dry_run mode.')
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


training_op = CustomCloudMLTrainingOperator(
    project_id='elvo-198322',
    task_id='train_model',
    dag=dag,
    mode='PRODUCTION'
)
