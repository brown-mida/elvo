import React, {Component} from 'react';
import ReactDOM from 'react-dom';

import Grid from '@material-ui/core/Grid'
import TextField from "@material-ui/core/TextField";
import Paper from "@material-ui/core/Paper/";
import axios from 'axios'

class App extends Component {
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
              <TextField
                  id="search"
                  label="Patient Id"
                  type="search"
                  onKeyPress={this.handleKeyPress}
                  margin="normal"
              />
            </Paper>
          </Grid>
          <Grid item sm={6}>
            <img src={'/static/540-559.png'}/>
          </Grid>
        </Grid>
    )
  }
}

ReactDOM.render(<App/>, document.getElementById('reactEntry'));