import React, {Component} from 'react';
import ReactDOM from 'react-dom';
import TextField from "@material-ui/core/TextField";
import Paper from "@material-ui/core/Paper/";
import axios from 'axios'
import Button from "@material-ui/core/Button";
import Drawer from '@material-ui/core/Drawer';

import PlaneSVG from './PlaneSVG';

const styles = {
  paper: {
    padding: '10px',
    paddingLeft: '240px',
  },
  dividerPaper: {
    padding: '10px',
    width: '400px',
  }
};

// TODO: Caching
class App extends Component {
  constructor(props) {
    super(props);

    let initialPatientId;
    let x1 = 100;
    let x2 = 200;
    let y1 = 100;
    let y2 = 200;
    let z1 = 100;
    let z2 = 200;

    if (App.getQueryVariable('patientId')) {
      console.log('loading state from query string');
      initialPatientId = App.getQueryVariable('patientId');
      x1 = App.getQueryVariable('x1');
      x2 = App.getQueryVariable('x2');
      y1 = App.getQueryVariable('y1');
      y2 = App.getQueryVariable('y2');
      z1 = App.getQueryVariable('z1');
      z2 = App.getQueryVariable('z2');
      console.log('initial state:', initialPatientId, x1, x2, y1, y2, z1, z2)
    }

    this.state = {
      patientId: initialPatientId,
      searchValue: initialPatientId,
      indices: {
        x: 125, // TODO: Rename to sagittalIndex
        y: 125, // TODO: Rename to coronalIndex
        z: 100, // TODO: Rename to axialIndex
        zMip: 100,
      },
      threshold: 120,
      dimensions: {
        z: 300,
        x: 250,
        y: 250,
      },
      roiDimensions: {
        x1: parseInt(x1),
        x2: parseInt(x2),
        y1: parseInt(y1),
        y2: parseInt(y2),
        z1: parseInt(z1),
        z2: parseInt(z2),
      },
      renderingParams: 'x1=100&x2=110&y1=100&y2=110&z1=100&z2=110',
      mipStep: 4,
      createdBy: '',
    };

    // Needed to bind this to class
    this.handleSearchKeyPress = this.handleSearchKeyPress.bind(this);
    this.getImageDimensions = this.getImageDimensions.bind(this);
    this.updateBoundingBox = this.updateBoundingBox.bind(this);
    this.handleAnnotation = this.handleAnnotation.bind(this);
    this.updateRenderingParams = this.updateRenderingParams.bind(this);
  }

  handleSearchKeyPress(event) {
    if (event.key === 'Enter') {
      console.log('enter');
      this.setState({patientId: this.state.searchValue});
      console.log('searching for patient:', this.state.patientId);
      this.getImageDimensions();
    }
  }

  static getQueryVariable(variable) {
    const query = window.location.search.substring(1);
    const vars = query.split("&");
    for (let i = 0; i < vars.length; i++) {
      const pair = vars[i].split("=");
      if (pair[0] === variable) {
        return pair[1];
      }
    }
    return (false);
  }

  static userGuide() {
    return (
        <Paper style={styles.dividerPaper}>
          <h2>User Guide</h2>
          <p>Enter patient ids from the <a
              href="https://docs.google.com/spreadsheets/d/1RVl0Zs3XtKEYSIK6vXsc5ZXonIwh1pJtpkDcPFKE3aw/edit#gid=0">
            Elvo Key
          </a> into the text box below
          </p>
          <p>Use the x1/y1/... fields to set the bounding box.</p>
          <p>To scroll, click on an input field to the right and use the up/down
            arrow
            keys.</p>
          <p>Uploaded annotations are listed in the <a
              href="https://docs.google.com/spreadsheets/d/1_j7mq_VypBxYRWA5Y7ef4mxXqU0EmBKDl0lkp62SsXA/edit#gid=0">
            Annotation spreasheet
          </a></p>
          <p><em>Only use this app on Wi-Fi</em></p>
        </Paper>
    )
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

  updateRenderingParams() {
    const renderingParams = (
        `x1=${this.state.roiDimensions.x1}&` +
        `x2=${this.state.roiDimensions.x2}&` +
        `y1=${this.state.roiDimensions.y1}&` +
        `y2=${this.state.roiDimensions.y2}&` +
        `z1=${this.state.roiDimensions.z1}&` +
        `z2=${this.state.roiDimensions.z2}`
    );
    this.setState({
      renderingParams: renderingParams,
    });
  }

  handleAnnotation() {
    if (this.state.createdBy === '') {
      console.error('Username cannot be empty');
      return;
    }
    const data = {
      created_by: this.state.createdBy,
      patient_id: this.state.patientId,
      x1: this.state.roiDimensions.x1,
      x2: this.state.roiDimensions.x2,
      y1: this.state.roiDimensions.y1,
      y2: this.state.roiDimensions.y2,
      z1: this.state.roiDimensions.z1,
      z2: this.state.roiDimensions.z2,
    };
    axios.post('/roi', data)
        .catch(() => {
          console.error('failed to insert annotation:', data);
        });
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

  annotationInputView() {
    return (
        <Paper style={styles.dividerPaper}>
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
              label="red1"
              margin="normal"
              type="number"
              value={this.state.roiDimensions.x1}
              onChange={this.updateBoundingBox('x1')}
          />
          <TextField
              id="x2"
              label="red2"
              margin="normal"
              type="number"
              value={this.state.roiDimensions.x2}
              onChange={this.updateBoundingBox('x2')}
          />
          <br/>
          <TextField
              id="y1"
              label="green1"
              margin="normal"
              type="number"
              value={this.state.roiDimensions.y1}
              onChange={this.updateBoundingBox('y1')}
          />
          <TextField
              id="y2"
              label="green2"
              margin="normal"
              type="number"
              value={this.state.roiDimensions.y2}
              onChange={this.updateBoundingBox('y2')}
          />
          <br/>
          <TextField
              id="z1"
              label="blue1"
              margin="normal"
              type="number"
              value={this.state.roiDimensions.z1}
              onChange={this.updateBoundingBox('z1')}
          />
          <TextField
              id="z2"
              label="blue2"
              margin="normal"
              type="number"
              value={this.state.roiDimensions.z2}
              onChange={this.updateBoundingBox('z2')}
          />
          <br/>
          <TextField
              id="created_by"
              label="Username"
              onChange={(event) => this.setState({
                createdBy: event.target.value,
              })}
              margin="normal"
              value={this.state.createdBy}
          />
          <Button
              variant="contained"
              onClick={this.handleAnnotation}
          >
            Create Annotation
          </Button>
        </Paper>
    )
  }

  render() {
    console.log('in render, state is:', this.state);
    return (
        <div>
          <Drawer
              variant="permanent"
              anchor="left"
          >
            {App.userGuide()}
            {this.annotationInputView()}
          </Drawer>
          <div style={{paddingLeft: 240}}>
            <Paper style={styles.paper}>
              <h2>Axial MIP</h2>
              <PlaneSVG viewType={'axial_mip'}
                        patientId={this.state.patientId}
                        width={this.state.dimensions.x}
                        height={this.state.dimensions.y}
                        colorX={'rgb(255, 0, 0)'}
                        colorY={'rgb(0, 255, 0)'}
                        roiX1={this.state.roiDimensions.x1}
                        roiX2={this.state.roiDimensions.x2}
                        roiY1={this.state.roiDimensions.y1}
                        roiY2={this.state.roiDimensions.y2}
                        posIndex={this.state.indices.zMip}
                        lineIndex={this.state.indices.y}
              />
              <div>
                <TextField
                    id="z"
                    label="blue axis (mip)"
                    margin="normal"
                    type="number"
                    inputProps={{step: this.state.mipStep}}
                    value={this.state.indices.zMip}
                    onChange={this.updateIndex('zMip')}
                />
              </div>
            </Paper>
            <Paper style={styles.paper}>
              <h2>Axial</h2>
              <PlaneSVG viewType={'axial'}
                        patientId={this.state.patientId}
                        width={this.state.dimensions.x}
                        height={this.state.dimensions.y}
                        colorX={'rgb(255, 0, 0)'}
                        colorY={'rgb(0, 255, 0)'}
                        roiX1={this.state.roiDimensions.x1}
                        roiX2={this.state.roiDimensions.x2}
                        roiY1={this.state.roiDimensions.y1}
                        roiY2={this.state.roiDimensions.y2}
                        posIndex={this.state.indices.z}
                        lineIndex={this.state.indices.y}
              />
              <div>
                <TextField
                    id="z"
                    label="blue axis"
                    margin="normal"
                    type="number"
                    value={this.state.indices.z}
                    onChange={this.updateIndex('z')}
                />
              </div>
            </Paper>
            <Paper style={styles.paper}>
              <h2>Coronal</h2>
              <PlaneSVG viewType={'coronal'}
                        patientId={this.state.patientId}
                        width={this.state.dimensions.x}
                        height={this.state.dimensions.z}
                        colorX={'rgb(255, 0, 0)'}
                        colorY={'rgb(0, 0, 255)'}
                        roiX1={this.state.roiDimensions.x1}
                        roiX2={this.state.roiDimensions.x2}
                        roiY1={this.state.roiDimensions.z1}
                        roiY2={this.state.roiDimensions.z2}
                        posIndex={this.state.indices.y}
                        lineIndex={this.state.indices.z}

              />
              <div>
                <TextField
                    id="y"
                    label="green axis"
                    margin="normal"
                    type="number"
                    value={this.state.indices.y}
                    onChange={this.updateIndex('y')}
                />
              </div>
            </Paper>
            <Paper style={styles.paper}>
              <h2>Sagittal</h2>
              <PlaneSVG viewType={'sagittal'} patientId={this.state.patientId}
                        width={this.state.dimensions.y}
                        height={this.state.dimensions.z}
                        colorX={'rgb(0, 255, 0)'}
                        colorY={'rgb(0, 0, 255)'}
                        roiX1={this.state.roiDimensions.y1}
                        roiX2={this.state.roiDimensions.y2}
                        roiY1={this.state.roiDimensions.z1}
                        roiY2={this.state.roiDimensions.z2}
                        posIndex={this.state.indices.x}
                        lineIndex={this.state.indices.z}
              />
              <div>
                <TextField
                    id="x"
                    label="red axis"
                    margin="normal"
                    type="number"
                    value={this.state.indices.x}
                    onChange={this.updateIndex('x')}
                />
              </div>
            </Paper>
            <Paper style={styles.paper}>
              <h2>3D</h2>
              <span>
                <img
                    src={`/image/rendering/${this.state.patientId}?${this.state.renderingParams}`}
                    style={{maxWidth: 300, maxHeight: 300}}
                />
                <Button
                    variant="contained"
                    onClick={this.updateRenderingParams}
                >
                  Update Rendering
                </Button>
                </span>
            </Paper>
          </div>
        </div>
    )
  }
}

ReactDOM.render(<App/>, document.getElementById('reactEntry'));