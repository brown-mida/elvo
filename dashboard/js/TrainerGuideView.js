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
          The model should finish training within 30 minutes. Note that only
          a single model may train at a time.
        </p>

        <p>
          After the model has finished training, you should be able to select your job from
          the plots dropdown and view plots of your model's effectiveness through
          the <b>Results</b> tab.
        </p>

        <p>
          The data dropdown gives you a selection of already-processed data that
          you can use to train the model. Select the
          <b> Data</b> tab to see samples of the processed data.
        </p>

        <p>
          If you would like to preprocess your own data, the preprocessing
          button allow you to choose a combination of different transformations
          to execute. A preprocessing job takes around 10 minutes to complete.
        </p>

        <h3>Options Description</h3>
        <h4>Preprocessed Data</h4>
        <p>The name you would like to give the data your generate with <b>Preprocess Data</b>. <em>It
          must be unique.</em>
          {/* TODO: Error */}
        </p>
        <h4>Crop Length</h4>
        <p>
          All images fed into a model must have fixed dimensions. A length of
          200 will crop the center 200 x 200 pixels, as shown below
        </p>
        <img src={'https://storage.googleapis.com/elvos-public/crop_length.png'}
             style={{ maxWidth: 200 }}
        />
        <h4>Mip Thickness</h4>
        The mip thickness represents the length between the two green lines
        below, in millimeters. The below example is a mip thickness of 25.
        <img src={'https://storage.googleapis.com/elvos-public/mip_thickness.png'}
             style={{ maxWidth: 200 }}
        />

        <p><em>Note that a 2D resnet will ingest <b>3</b> slices.</em> The default thickness and
          offset combination will cover eye-height as well.
        </p>

        <h4>Height Offset</h4>
        The distance (in millimeters) from the top of the image to start building
        a MIP. The above image has a height offset of 30.

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