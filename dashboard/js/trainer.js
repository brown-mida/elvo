import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';

import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import FormControl from '@material-ui/core/FormControl';
import Drawer from '@material-ui/core/Drawer';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';


const styles = {
  inputField: {
    margin: 10,
  },
  grid: {
    paddingLeft: 500,
  },
  plotImg: {
    maxWidth: '60%',
  },
};

class Trainer extends Component {

  constructor(props) {
    super(props);

    this.state = {
      dataName: 'processed-lower',
      authorName: 'web-luke',
      jobName: 'web-job',
      modelName: 'resnet',
      // Keep the hyper-parameters as strings and use the material-ui
      // components to validate the input.
      valSplit: '0.1',
      // TODO(luke): Validate as integer but not float
      batchSize: '8',
      maxEpochs: '70',

      allPlots: [],
      selectedPlot: '',
    };

    this.handleChange = this.handleChange.bind(this);
    this.sendJobRequest = this.sendJobRequest.bind(this);
    this.evalView = this.evalView.bind(this);
  }

  componentDidMount() {
    // Set allPlots
    axios.get('/plots')
        .then(response => {
          this.setState({
            allPlots: response.data,
          });
        })
        .catch(error => {
          console.error(error);
        });
  }

  sendJobRequest() {
    const data = this.state;
    axios.post('/model', data)
        .then(response => {
          console.log(response);
        })
        .catch(error => {
          console.error(error);
        });
  }

  handleChange(name) {
    return event => {
      this.setState({
        [name]: event.target.value,
      });
    };
  }

  plotUrl(jobWithDate, plotType) {
    const url = 'https://storage.googleapis.com/elvos-public/plots/' +
        jobWithDate + '/' + plotType + '.png';
    return url;
  }

  // Returns the training view
  trainView() {
    // TODO(luke)
  }

  // Returns the evaluation view
  evalView() {
    return (
        <Grid container spacing={8} style={styles.grid}>
          <Grid item xs={12}>
            <Paper>
              <img src={this.plotUrl(this.state.selectedPlot, 'loss')}
                   style={styles.plotImg}
              />
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper>
              <img src={this.plotUrl(this.state.selectedPlot, 'acc')}
                   style={styles.plotImg}
              />
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper>
              <img src={this.plotUrl(this.state.selectedPlot, 'cm')}
                   style={styles.plotImg}
              />
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper>
              <h4>True Positives</h4>
              <img
                  src={
                    this.plotUrl(this.state.selectedPlot, 'true_positives')}
                  style={styles.plotImg}
              />
            </Paper>
          </Grid>

          <Grid item xs={12}>
            <Paper>
              <h4>False Positives</h4>
              <img src={
                this.plotUrl(this.state.selectedPlot, 'false_positives')}
                   style={styles.plotImg}
              />
            </Paper>
          </Grid>


          <Grid item xs={12}>
            <h4>True Negatives</h4>
            <Paper>
              <img src={
                this.plotUrl(this.state.selectedPlot, 'true_negatives')}
                   style={styles.plotImg}
              />
            </Paper>
          </Grid>


          <Grid item xs={12}>
            <Paper>
              <h4>False Negatives</h4>
              <img src={
                this.plotUrl(this.state.selectedPlot, 'false_negatives')}
                   style={styles.plotImg}
              />
            </Paper>
          </Grid>
        </Grid>
    );
  }

  render() {
    // TODO: Descriptions of the different fields.
    console.log('state', this.state);

    // Also accept the empty option as a default
    const plotOptions = [<option key={''} value={''}>{''}</option>];

    this.state.allPlots.forEach((e) => {
      plotOptions.push(<option key={e} value={e}>{e}</option>);
    });

    return (
        <div>
          {/* Start of the sidebar */}
          <Drawer
              variant="permanent"
              anchor="left"
          >
            <TextField
                id="jobName"
                label={'Job Name'}
                value={this.state.jobName}
                onChange={this.handleChange('jobName')}
                margin="normal"
                style={styles.inputField}
            />

            <TextField
                id="authorName"
                label={'Your Name'}
                value={this.state.authorName}
                onChange={this.handleChange('authorName')}
                margin="normal"
                style={styles.inputField}
            />
            <br/>

            <FormControl style={styles.inputField}>
              <InputLabel>Data</InputLabel>
              <Select
                  native
                  value={this.state.dataName}
                  onChange={this.handleChange('dataName')}
              >
                {/* TODO(luke): Make this a list element. Find a way to keep
               this in sync with what's available on GCS (perhaps a metadata
               db)*/}
                <option value={'processed-lower'}>processed-lower</option>
                <option value={'processed-lower-nbv'}>processed-lower-nbv</option>
                <option value={'processed-lower-no-vert'}>processed-lower-no-vert</option>
                <option value={'processed-no-basvert'}>processed-no-basvert</option>
              </Select>
            </FormControl>

            <FormControl style={styles.inputField}>
              <InputLabel>Model</InputLabel>
              <Select
                  native
                  value={this.state.modelName}
                  onChange={this.handleChange('modelName')}
              >
                <option value={'resnet'}>ResNet</option>
              </Select>
            </FormControl>

            {/*<TextField*/}
            {/*id="batchSize"*/}
            {/*label={'Batch Size'}*/}
            {/*value={this.state.batchSize}*/}
            {/*onChange={this.handleChange('batchSize')}*/}
            {/*margin="normal"*/}
            {/*type="number"*/}
            {/*/>*/}

            <br/>
            <br/>

            <Button
                variant="outlined"
                onClick={this.sendJobRequest}
            >
              Create Training Job
            </Button>

            <FormControl style={styles.inputField}>
              <InputLabel>Plots</InputLabel>
              <Select
                  native
                  value={this.state.selectedPlot}
                  onChange={this.handleChange('selectedPlot')}
              >
                {plotOptions}
              </Select>
            </FormControl>
          </Drawer>

          {/* Start of the main body */}
          {this.evalView()}
        </div>
    );
  }
}

ReactDOM.render(<Trainer/>, document.getElementById('reactEntry'));
