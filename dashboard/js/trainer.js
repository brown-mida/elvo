import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import axios from 'axios';

import Button from '@material-ui/core/Button';

class Trainer extends Component {
  sendJobRequest() {
    const data = {
      model: 'resnet',
    };

    axios.post('/model/add', data)
        .then(response => {
          console.log(response);
        })
        .catch(error => {
          console.error(error);
        });
  }

  render() {
    return (
        <Button
            variant="outlined"
            onClick={this.sendJobRequest}
        >
          Create Training Job
        </Button>
    );
  }
}

ReactDOM.render(<Trainer/>, document.getElementById('reactEntry'));
