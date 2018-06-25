import React, {Component} from 'react';
import ReactDOM from 'react-dom';

import Grid from '@material-ui/core/Grid'
import TextField from "@material-ui/core/TextField";
import Paper from "@material-ui/core/Paper/";
import axios from 'axios'
import Button from "@material-ui/core/Button";

const styles = {
  paper: {
    padding: '10px',
  }
};

// TODO: Caching
class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      patientId: '0DQO9A6UXUQHR8RA',
      searchValue: '0DQO9A6UXUQHR8RA',
      indices: {
        x: 125, // TODO: Rename to sagittalIndex
        y: 0, // TODO: Rename to coronalIndex
        z: 230, // TODO: Rename to axialIndex
      },
      threshold: 120,
      dimensions: {
        z: 300,
        x: 250,
        y: 250,
      },
      roiDimensions: {
        x1: 100,
        x2: 200,
        y1: 100,
        y2: 200,
        z1: 100,
        z2: 200,
      },
      step: 4,
    };

    // Needed to bind this to class
    this.handleSearchKeyPress = this.handleSearchKeyPress.bind(this);
    this.getImageDimensions = this.getImageDimensions.bind(this);
    this.updateBoundingBox = this.updateBoundingBox.bind(this);
    this.updateImageCache = this.updateImageCache.bind(this);
  }

  handleSearchKeyPress(event) {
    if (event.key === 'Enter') {
      console.log('enter');
      this.setState({patientId: this.state.searchValue});
      console.log('searching for patient:', this.state.patientId);
      this.getImageDimensions();
    }
  }

  updateBoundingBox(attr) {
    return (event) => {
      const value = parseInt(event.target.value);
      if (!isNaN(value)) {
        this.setState((state) => {
          state.roiDimensions[attr] = value;
          return state;
        })
      }
    }
  }

  updateIndex(attr) {
    return (event) => {
      const value = parseInt(event.target.value);
      if (!isNaN(value)) {
        this.setState((state) => {
          state.indices[attr] = value;
          return state;
        })
      }
    }
  }

  updateImageCache() {
    console.log('updating cache');
    const imageElements = [];
    for (let i = Math.max(0, this.state.indices.x + i - 5);
         i <= Math.min(this.state.dimensions.x, this.state.indices.x + 5);
         i += this.state.step) {
      imageElements.push(<image
          href={`/image/sagittal/${this.state.patientId}/${i}`}/>);
    }
    for (let i = Math.max(0, this.state.indices.z + i - 5);
         i <= Math.min(this.state.dimensions.z, this.state.indices.z + 5);
         i += this.state.step) {
      imageElements.push(<image
          href={`/image/axial/${this.state.patientId}/${i}`}/>);
      imageElements.push(<image
          href={`/image/axial/${this.state.patientId}/${i}`}/>);
    }
    return imageElements;
  }

  getImageDimensions() {
    self = this;
    axios.get(`/image/dimensions/${this.state.patientId}`)
        .then((response) => {
          console.log('response', response);
          self.setState(prevState => {
            prevState.dimensions = {
              z: response.data.z,
              x: response.data.x,
              y: response.data.y,
            }
          })
        })
        .catch((error) => {
          console.error(error);
        })
  }

  render() {
    console.log('in render, state is:', this.state);
    const imagesToCache = this.updateImageCache();
    return (
      <Grid container spacing={16}>
        <Grid item sm={6}>
          <Paper style={styles.paper}>
            <h2>User Guide</h2>
            <p>Use the text fields below to set the bounding box.</p>
            <p>To scroll, click on an input field and use the up/down arrow
              keys.</p>
            <p><em>Only use this app on Wi-Fi</em></p>
            <h2>To Do</h2>
            <p>Improve scrolling performance w/ image caching</p>
          </Paper>
        </Grid>
        <Grid item sm={6}>
          <Paper style={styles.paper}>
            <h2>Sagittal</h2>
            <svg height={this.state.dimensions.z}
                 width={this.state.dimensions.y}
            >
              <image
                  href={`/image/sagittal/${this.state.patientId}/${this.state.indices.x}`}/>
              <line x1="0" y1={this.state.dimensions.z - this.state.indices.z}
                    x2={this.state.dimensions.y}
                    y2={this.state.dimensions.z - this.state.indices.z}
                    style={{
                      stroke: 'rgb(255, 255, 255)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.y1}
                    y1={this.state.roiDimensions.z2}
                    x2={this.state.roiDimensions.y2}
                    y2={this.state.roiDimensions.z2}
                    style={{
                      stroke: 'rgb(0, 255, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.y1}
                    y1={this.state.roiDimensions.z1}
                    x2={this.state.roiDimensions.y2}
                    y2={this.state.roiDimensions.z1}
                    style={{
                      stroke: 'rgb(0, 255, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.y1}
                    y1={this.state.roiDimensions.z1}
                    x2={this.state.roiDimensions.y1}
                    y2={this.state.roiDimensions.z2}
                    style={{
                      stroke: 'rgb(0, 0, 255)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.y2}
                    y1={this.state.roiDimensions.z1}
                    x2={this.state.roiDimensions.y2}
                    y2={this.state.roiDimensions.z2}
                    style={{
                      stroke: 'rgb(0, 0, 255)',
                      strokeWidth: 2,
                    }}
              />
            </svg>
            <div>
              <TextField
                  id="x"
                  label="x"
                  margin="normal"
                  type="number"
                  inputProps={{step: this.state.step}}
                  value={this.state.indices.x}
                  onChange={this.updateIndex('x')}
              />
            </div>
          </Paper>
        </Grid>
        <Grid item sm={6}>
          <Paper style={styles.paper}>
            <TextField
                id="search"
                label="Patient Id"
                type="search"
                onChange={(event) => this.setState({
                  searchValue: event.target.value,
                })}
                onKeyPress={this.handleSearchKeyPress}
                margin="normal"
                value={this.state.searchValue}
            />
            <br/>
            <TextField
                id="x1"
                label="x1"
                margin="normal"
                value={this.state.roiDimensions.x1}
                onChange={this.updateBoundingBox('x1')}
            />
            <TextField
                id="x2"
                label="x2"
                margin="normal"
                value={this.state.roiDimensions.x2}
                onChange={this.updateBoundingBox('x2')}
            />
            <br/>
            <TextField
                id="y1"
                label="y1"
                margin="normal"
                value={this.state.roiDimensions.y1}
                onChange={this.updateBoundingBox('y1')}
            />
            <TextField
                id="y2"
                label="y2"
                margin="normal"
                value={this.state.roiDimensions.y2}
                onChange={this.updateBoundingBox('y2')}
            />
            <br/>
            <TextField
                id="z1"
                label="z1"
                margin="normal"
                value={this.state.roiDimensions.z1}
                onChange={this.updateBoundingBox('z1')}
            />
            <TextField
                id="z2"
                label="z2"
                margin="normal"
                value={this.state.roiDimensions.z2}
                onChange={this.updateBoundingBox('z2')}
            />
            <br/>
            <Button
                variant="contained"
            >
              Update
            </Button>
          </Paper>
        </Grid>
        <Grid item sm={6}>
          <Paper style={styles.paper}>
            <h2>Axial</h2>
            <svg height={this.state.dimensions.x}
                 width={this.state.dimensions.y}
            >
              <image
                  href={`/image/axial/${this.state.patientId}/${this.state.indices.z}`}/>
              <line x1={this.state.roiDimensions.x1}
                    y1={this.state.roiDimensions.y1}
                    x2={this.state.roiDimensions.x2}
                    y2={this.state.roiDimensions.y1}
                    style={{
                      stroke: 'rgb(255, 0, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.x1}
                    y1={this.state.roiDimensions.y2}
                    x2={this.state.roiDimensions.x2}
                    y2={this.state.roiDimensions.y2}
                    style={{
                      stroke: 'rgb(255, 0, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.x1}
                    y1={this.state.roiDimensions.y1}
                    x2={this.state.roiDimensions.x1}
                    y2={this.state.roiDimensions.y2}
                    style={{
                      stroke: 'rgb(0, 255, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.x2}
                    y1={this.state.roiDimensions.y1}
                    x2={this.state.roiDimensions.x2}
                    y2={this.state.roiDimensions.y2}
                    style={{
                      stroke: 'rgb(0, 255, 0)',
                      strokeWidth: 2,
                    }}
              />
            </svg>
            <div>
              <TextField
                  id="z"
                  label="z"
                  margin="normal"
                  inputProps={{step: this.state.step}}
                  type="number"
                  value={this.state.indices.z}
                  onChange={this.updateIndex('z')}
              />
            </div>
          </Paper>
        </Grid>
        <Grid item sm={6}>
          <Paper style={styles.paper}>
            <h2>3D</h2>
            <span>
              {/*<img*/}
              {/*src={`/image/rendering/${this.state.patientId}/${this.state.threshold}`}*/}
              {/*style={{maxWidth: 300, maxHeight: 300}}*/}
              {/*/>*/}
              {/*<Button> /!* 'TODO: Actually re-render with ROI zoom' *!/*/}
              {/*Re-Render*/}
              {/*</Button>*/}
              3D View Not Ready Yet
              </span>
          </Paper>
        </Grid>
        <Grid item sm={6}>
          <Paper style={styles.paper}>
            <h2>Axial MIP</h2>
            <svg height={this.state.dimensions.x}
                 width={this.state.dimensions.y}
            >
              <image
                  href={`/image/axial_mip/${this.state.patientId}/${this.state.indices.z}`}/>
              <line x1={this.state.roiDimensions.x1}
                    y1={this.state.roiDimensions.y1}
                    x2={this.state.roiDimensions.x2}
                    y2={this.state.roiDimensions.y1}
                    style={{
                      stroke: 'rgb(255, 0, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.x1}
                    y1={this.state.roiDimensions.y2}
                    x2={this.state.roiDimensions.x2}
                    y2={this.state.roiDimensions.y2}
                    style={{
                      stroke: 'rgb(255, 0, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.x1}
                    y1={this.state.roiDimensions.y1}
                    x2={this.state.roiDimensions.x1}
                    y2={this.state.roiDimensions.y2}
                    style={{
                      stroke: 'rgb(0, 255, 0)',
                      strokeWidth: 2,
                    }}
              />
              <line x1={this.state.roiDimensions.x2}
                    y1={this.state.roiDimensions.y1}
                    x2={this.state.roiDimensions.x2}
                    y2={this.state.roiDimensions.y2}
                    style={{
                      stroke: 'rgb(0, 255, 0)',
                      strokeWidth: 2,
                    }}
              />
            </svg>
            <div>
              <TextField
                  id="z"
                  label="z"
                  margin="normal"
                  type="number"
                  inputProps={{step: this.state.step}}
                  value={this.state.indices.z}
                  onChange={this.updateIndex('z')}
              />
            </div>
          </Paper>
        </Grid>
        <div style={{display: 'hidden'}}>
          {imagesToCache}
        </div>
      </Grid>
    )
  }
}

ReactDOM.render(
    <App/>
    , document.getElementById('reactEntry'));
