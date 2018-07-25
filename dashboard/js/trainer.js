import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';

import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import FormControl from '@material-ui/core/FormControl';
import Drawer from '@material-ui/core/Drawer';
import BottomNavigation from '@material-ui/core/BottomNavigation';
import BottomNavigationAction from '@material-ui/core/BottomNavigationAction';
import Divider from '@material-ui/core/Divider';
import TrainerResultsView from './TrainerResultsView';
import TrainerDataView from './TrainerDataView';
import TrainerGuideView from './TrainerGuideView';


const styles = {
  inputField: {
    margin: 10,
  },
  grid: {
    paddingLeft: 500,
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

      allDataNames: [],
      imageNames: [],
      offset: 0,

      allPlots: [],
      selectedPlot: '',
      plotSortType: 'date',

      viewType: 'guide',
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleDataChange = this.handleDataChange.bind(this);
    this.sendJobRequest = this.sendJobRequest.bind(this);
  }

  componentDidMount() {
    // Set allPlots and selectedPLot
    axios.get('/plots')
        .then(response => {
          this.setState({
            allPlots: response.data,
            selectedPlot: response.data[0],
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
            dateName: response.data[0],
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

  render() {
    // TODO: Descriptions of the different fields.
    console.log('state', this.state);

    const dataOptions = this.state.allDataNames.map(name => {
      return <option key={name} value={name}>{name}</option>;
    });

    const sortedAllPlots = this.state.allPlots
        .slice()
        .sort((a, b) => sortByDate(a, b, false));
    const plotOptions = [];
    sortedAllPlots.forEach((e) => {
      plotOptions.push(<option key={e} value={e}>{e}</option>);
    });

    let bodyView;
    switch (this.state.viewType) {
      case 'data':
        bodyView = (
            <TrainerDataView
                dataName={this.state.dataName}
                imageNames={this.state.imageNames}
                offset={this.state.offset}
                parentStyles={styles}
            />
        );
        break;
      case 'results':
        bodyView = (
            <TrainerResultsView
                selectedPlot={this.state.selectedPlot}
                parentStyles={styles}
            />
        );
        break;
      case 'guide':
        bodyView = (
            <TrainerGuideView parentStyles={styles}/>
        );
        break;
      default:
        console.error(this.state.viewType + ' is not valid');
        bodyView = <div>Error :(</div>;
    }

    return (
        <div>
          {/* Start of the sidebar */}
          <Drawer
              variant="permanent"
              anchor="left"
          >
            <h3 style={{ paddingLeft: 10 }}>Training Options</h3>

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

            <Divider/>

            <h3 style={{ paddingLeft: 10 }}>Results Options</h3>

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

            <Button href={'http://104.196.51.205:5601/'}>Kibana</Button>

            <Divider/>

            <h3 style={{ paddingLeft: 10 }}>View</h3>

            <BottomNavigation
                value={this.state.viewType}
                onChange={(event, viewType) => {
                  this.setState({ viewType });
                }}
                showLabels
            >
              <BottomNavigationAction label="Data" value="data"/>
              <BottomNavigationAction label="Results" value="results"/>
              <BottomNavigationAction label="Guide" value="guide"/>
            </BottomNavigation>
          </Drawer>
          {bodyView}
        </div>
    );
  }
}

// TODO(luke): This won't work in 2020. This kind of code appears
// in a few other places as well.
function sortByDate(a, b, ascending = true) {
  const aDateIndex = a.indexOf('201');
  const bDateIndex = b.indexOf('201');
  const aDate = a.slice(aDateIndex);
  const bDate = b.slice(bDateIndex);

  const sign = (ascending === true) ? 1 : -1;

  if (aDate < bDate) {
    return -sign;
  }

  if (aDate > bDate) {
    return sign;
  }

  return 0;
}

ReactDOM.render(<Trainer/>, document.getElementById('reactEntry'));
