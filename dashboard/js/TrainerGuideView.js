import React from 'react';

const TrainerGuideView = ({ parentStyles }) => {
  return (
      <div style={parentStyles.grid}>
        <h2>User Guide</h2>
        <p>This app allows you to train deep learning models on the ELVO data
          and view the results.</p>

        <p>To train a model, click on the CREATE TRAINING JOB button and then
          go to <a href="http://104.196.51.205:8080/admin/">http://104.196.51.205:8080/admin/</a>.

          {/* TODO(luke): Remove this when the completion/heartbeak link is made.*/}
          If there is a train_model DAG run in the "running" state, then the job has been created.
          The model should finish training within 30 minutes.
        </p>

        <p>
          After the model has finished training, you should be able to select your job from
          the plots dropdown and view plots of your model's effectiveness through
          the <b>Results</b> tab.
        </p>

        <p>
          The data dropdown gives you a selection of already-processed data that
          you can use to train the model. Currently these are all MIPs, in the
          future, we hope to allow you to define your preprocessing functions
          so you can fine-tune that step before building a job. Select the
          <b>Data</b> tab to see samples of the processed data.
        </p>

        <h3>Options Description</h3>
        <h4>Job Name</h4>
        <p>The name of your job, used to help you identify the models you train</p>
        <h4>Data</h4>
        <p>The type of processed data you would like to use to train your model</p>
        <h4>Model</h4>
        <p>The model type to train. Currently only 2D binary
          classification (ResNet) is supported.
          ELVO location detection and 3D classification will be supported in the future.
        </p>
        <h4>Plots</h4>
        <p>The training job to show plots of in the <b>Results</b> view</p>
        <h4>Kibana</h4>
        <p>A more detailed dashboard of the model results, allowing you to see
          advanced metrics and compare your job with past runs</p>
      </div>
  );
};

export default TrainerGuideView;