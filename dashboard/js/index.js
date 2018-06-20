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
      searchValue: '',
      axialIndex: 230,
      axialMipIndex: 23,
      sagittalIndex: 125,
      threshold: 120,
      dimensions: {
        i: 300,
        j: 250,
        k: 250,
      },
      roiDimensions: {
        x1: 0,
        x2: 0,
        y1: 0,
        y2: 0,
        z1: 0,
        z2: 0,
      }
    };

    // Needed to bind this to class
    this.handleKeyPress = this.handleKeyPress.bind(this);
    this.getImageDimensions = this.getImageDimensions.bind(this);
  }

  handleKeyPress(event) {
    if (event.key === 'Enter') {
      console.log('enter');
      this.setState({patientId: this.state.searchValue});
      console.log('searching for patient:', this.state.patientId);
      this.getImageDimensions();
    }
  }

  getImageDimensions() {
    self = this;
    axios.get(`/image/dimensions/${this.state.patientId}`)
        .then((response) => {
          console.log('response', response);
          self.setState(prevState => {
            prevState['dimensions'] = {
              i: response.data['i'],
              j: response.data['j'],
              k: response.data['k'],
            }
          })
        })
        .catch((error) => {
          console.error(error);
        })
  }

  render() {
    console.log('in render, state is:', this.state);
    return (
        <Grid container spacing={16}>
          <Grid item sm={6}>
            <Paper>
              <h2>Saggital</h2>
              <svg height={this.state.dimensions['i']}
                   width={this.state.dimensions['k']}
                   style={{
                     zIndex: 1,
                     backgroundImage: `url(/image/sagittal/${this.state.patientId}/${this.state.sagittalIndex})`
                   }}
              >
                <line x1="0" y1={this.state.axialMipIndex}
                      x2={this.state.dimensions['k']}
                      y2={this.state.axialMipIndex}
                      style={{
                        stroke: 'rgb(255, 0, 0)',
                        strokeWidth: 2,
                      }}
                />
              </svg>
            </Paper>
          </Grid>
          <Grid item sm={6}>
            <Paper>
              <h2>Axial</h2>
              <svg height={this.state.dimensions['j']}
                   width={this.state.dimensions['k']}
                   style={{
                     zIndex: 1,
                     backgroundImage: `url(/image/axial/${this.state.patientId}/${this.state.axialIndex})`
                   }}
              >
                <line x1="0" y1="0"
                      x2="200" y2="200"
                      style={{
                        stroke: 'rgb(255, 0, 0)',
                        strokeWidth: 2,
                      }}
                />
              </svg>
            </Paper>
          </Grid>
          {/*<Grid item sm={6}>*/}
          {/*<Paper>*/}
          {/*<h2>3D</h2>*/}
          {/*<span>*/}
          {/*<img*/}
          {/*src={`/image/rendering/${this.state.patientId}/${this.state.threshold}`}*/}
          {/*style={{maxWidth: 300, maxHeight: 300}}*/}
          {/*/>*/}
          {/*<Button> {'TODO: Actually re-render with ROI zoom'}*/}
          {/*Re-Render*/}
          {/*</Button>*/}
          {/*</span>*/}
          {/*</Paper>*/}
          {/*</Grid>*/}
          {/*<Grid item sm={6}>*/}
          {/*<Paper>*/}
          {/*<h2>Axial MIP</h2>*/}
          {/*<svg height={this.state.dimensions['j']}*/}
          {/*width={this.state.dimensions['k']}*/}
          {/*style={{*/}
          {/*zIndex: 1,*/}
          {/*backgroundImage: `url(/image/axial_mip/${this.state.patientId}/${this.state.axialMipIndex})`*/}
          {/*}}*/}
          {/*>*/}
          {/*<line x1="0" y1="0"*/}
          {/*x2="200" y2="200"*/}
          {/*style={{*/}
          {/*stroke: 'rgb(255, 0, 0)',*/}
          {/*strokeWidth: 2,*/}
          {/*}}*/}
          {/*/>*/}
          {/*</svg>*/}
          {/*</Paper>*/}
          {/*</Grid>*/}
          <Grid item sm={6}>
            <Paper>
              <TextField
                  id="search"
                  label="Patient Id"
                  type="search"
                  onChange={(event) => this.setState({
                    searchValue: event.target.value,
                  })}
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
                  value={this.state.searchValue}
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