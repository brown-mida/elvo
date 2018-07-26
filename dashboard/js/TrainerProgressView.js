import React, { Component } from 'react';
import axios from 'axios';

const styles = {
  iframe: { minWidth: '70vw', minHeight: '60vh' },
};

const sendCountRequest = () => {
  axios.get('/');
};

class TrainerProgressView extends Component {

  constructor(props) {
    super(props);
    this.state = {
      count: -1,
      intervalId: null,
      inProgress: true,
    };
  }

  componentDidMount() {
    this.sendCountRequest();
    const intervalId = setInterval(() => {
      this.sendCountRequest();
    }, 30 * 1000);
    this.setState({
      intervalId,
    });
  }

  // Sends a request and updates the count state. If the count
  // has not changed, stop the progress bar.
  sendCountRequest() {
    axios.get(`/preprocessing/${this.props.processedName}/count`)
        .then(((response) => {
          const count = response.data.count;

          if (count === this.state.count) {
            // TODO: Consider clearing this interval.
            // clearInterval(this.state.intervalId);
            this.setState({
              intervalId: null,
              inProgress: false,
            });
          }

          this.setState({
            count,
          });
        }))
        .catch((error) => console.log(error));
  }

  render() {
    return (
        <div style={this.props.parentStyles.grid}>
          <h3>Preprocessing Jobs</h3>
          <h4>Number of arrays in <b>{this.props.processedName}</b></h4>
          {this.state.count}

          <h4>Recent Jobs</h4>
          <iframe
              src="http://104.196.51.205:8080/admin/dagrun/?flt1_dag_id_equals=preprocess_web"
              style={styles.iframe}
          />
          <h3>Training</h3>

          <h4>Recent Jobs</h4>
          <iframe
              src="http://104.196.51.205:8080/admin/dagrun/?flt1_dag_id_equals=train_model"
              style={styles.iframe}
          />
        </div>
    );
  }
}

export default TrainerProgressView;