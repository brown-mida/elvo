import React, {Component} from 'react';
import ReactDOM from 'react-dom';

import Grid from '@material-ui/core/Grid'
import TextField from "@material-ui/core/TextField";
import Paper from "@material-ui/core/Paper/";
import axios from 'axios'

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      patientId: '0DQO9A6UXUQHR8RA',
      axialIndex: 230,
      axialMipIndex: 23,
      sagittalIndex: 125,
      threshold: 120,
    }
  }

  handleKeyPress(event) {
    if (event.key === 'Enter') {
      this.fetchPatient(event.target.value);
    }
  }

  fetchPatient(id) {
    // TODO: Put data into elvos-public
    axios.get(`http://storage.googleapis.com/elvos-public/png/{id}`)
        .then((res) => {
          console.log('worked!')
        })
        .then((err) => {
          console.error(err)
        })
  }

  render() {
    console.log('render');
    return (
        <Grid container spacing={16}>
          <Grid item sm={6}>
            <Paper>
              <img
                  src={`/image/sagittal/${this.state.patientId}/${this.state.sagittalIndex}`}
              />
            </Paper>
          </Grid>
          <Grid item sm={6}>
            <Paper>
              <img
                  src={`/image/axial/${this.state.patientId}/${this.state.axialIndex}`}
              />
            </Paper>
          </Grid>
          <Grid item sm={6}>
            <Paper>
              <img
                  src={`/image/rendering/${this.state.patientId}/${this.state.threshold}`}
                  style={{maxWidth: 300, maxHeight: 300}}
              />
            </Paper>
          </Grid>
          <Grid item sm={6}>
            <Paper>
              <img
                  src={`/image/axial_mip/${this.state.patientId}/${this.state.axialMipIndex}`}
                  style={{maxWidth: 300, maxHeight: 300}}
              />
            </Paper>
          </Grid>
          <Grid item sm={6}>
            <Paper>
              <TextField
                  id="search"
                  label="Patient Id"
                  type="search"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
              <br/>
              <TextField
                  id="x1"
                  label="x1"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
              <TextField
                  id="x2"
                  label="x2"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
              <br/>
              <TextField
                  id="y1"
                  label="y1"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
              <TextField
                  id="y2"
                  label="y2"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
              <br/>
              <TextField
                  id="z1"
                  label="z1"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
              <TextField
                  id="z1"
                  label="z1"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
            </Paper>
          </Grid>
        </Grid>
    )
  }
}

ReactDOM.render(
    <App/>
    , document.getElementById('reactEntry'));