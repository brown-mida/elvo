import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';

import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import FormControl from '@material-ui/core/FormControl';

const DIR_PATH = '/home/lzhu7/elvo-analysis/data';

const styles = {
  inputField: {
    margin: 10,
  },
};

class Trainer extends Component {

  constructor(props) {
    super(props);

    this.state = {
      dataName: 'processed-lower',
      authorName: 'webbie',
      jobName: 'my-job',
      modelName: 'resnet',
      valSplit: '0.1',
      batchSize: '8',
      maxEpochs: '70',
    };

    this.handleChange = this.handleChange.bind(this);
    this.sendJobRequest = this.sendJobRequest.bind(this);
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

  render() {
    console.log('state', this.state);
    return (
        <div>
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
        </div>
    );
  }
}

ReactDOM.render(<Trainer/>, document.getElementById('reactEntry'));
