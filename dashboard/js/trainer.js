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
import BottomNavigation from '@material-ui/core/BottomNavigation';
import BottomNavigationAction from '@material-ui/core/BottomNavigationAction';


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
  dataImg: {
    maxWidth: 64,
  },
};

class Trainer extends Component {

  constructor(props) {
    super(props);

    this.state = {
      dataName: '',
      authorName: 'web-luke',
      jobName: 'web-job',
      modelName: 'resnet',
      // Keep the hyper-parameters as strings and use the material-ui
      // components to validate the input.
      valSplit: '0.1',
      // TODO(luke): Validate as integer but not float
      batchSize: '8',
      maxEpochs: '70',

      allDataNames: [],
      imageNames: [],
      offset: 0,

      allPlots: [],
      selectedPlot: '',

      viewType: 'data',
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleDataChange = this.handleDataChange.bind(this);
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

    // Set allDataNames
    axios.get('/data')
        .then(response => {
          this.setState({
            allDataNames: response.data,
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

  // Handles general changes in the input fields
  handleChange(name) {
    return event => {
      this.setState({
        [name]: event.target.value,
      });
    };
  }

  // When the  data is changed, ImageURLs is also updated
  handleDataChange(event) {
    const dataName = event.target.value;
    this.setState({
      dataName,
    });
    // Update imageNames
    axios.get('/data/' + dataName)
        .then(response => {
          this.setState({
            imageNames: response.data,
          });
        })
        .catch((error) => {
          console.error(error);
        });
  }

  plotUrl(jobWithDate, plotType) {
    const url = 'https://storage.googleapis.com/elvos-public/plots/' +
        jobWithDate + '/' + plotType + '.png';
    return url;
  }

  // Returns the training view
  trainView() {
    const baseURL = 'https://storage.googleapis.com/elvos-public/processed';
    const images = this.state.imageNames
        .slice(this.state.offset, this.state.offset + 32)
        .map((name) => {
          return (
              <Grid item xs={4}>
                <img src={`${baseURL}/${this.state.dataName}/arrays/${name}`}/>
              </Grid>);
        });
    return (
        <Grid container spacing={8} style={styles.grid}>
          {images}
        </Grid>
    );
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

    const dataOptions = this.state.allDataNames.map(name => {
      return <option key={name} value={name}>{name}</option>;
    });

    dataOptions.unshift(<option key={''} value={''}>{''}</option>);

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
                  onChange={this.handleDataChange}
              >
                {dataOptions}
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

            {/*TODO: Divider*/}

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

            <BottomNavigation
                value={this.state.viewType}
                onChange={(event, viewType) => {
                  this.setState({ viewType });
                }}
                showLabels
            >
              <BottomNavigationAction label="Data View" value="data"/>
              <BottomNavigationAction label="Results View" value="results"/>
            </BottomNavigation>
          </Drawer>

          {/* Start of the main body */}
          {this.state.viewType === 'data' ? this.trainView() : this.evalView()}
        </div>
    );
  }
}

ReactDOM.render(<Trainer/>, document.getElementById('reactEntry'));
