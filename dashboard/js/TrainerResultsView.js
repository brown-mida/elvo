import React from 'react';
import Grid from '@material-ui/core/Grid';
import Paper from '@material-ui/core/Paper';

const styles = {
  plotImg: {
    maxWidth: '60%',
  },
};

const plotUrl = (jobWithDate, plotType) => {
  return 'https://storage.googleapis.com/elvos-public/plots/' +
      jobWithDate + '/' + plotType + '.png';
};

const TrainerResultsView = ({ selectedPlot, parentStyles }) => {
  return (<Grid container spacing={8} style={parentStyles.grid}>
    <Grid item xs={12}>
      <Paper>
        <img src={plotUrl(selectedPlot, 'loss')}
             style={styles.plotImg}
        />
      </Paper>
    </Grid>

    <Grid item xs={12}>
      <Paper>
        <img src={plotUrl(selectedPlot, 'acc')}
             style={styles.plotImg}
        />
      </Paper>
    </Grid>

    <Grid item xs={12}>
      <Paper>
        <img src={plotUrl(selectedPlot, 'cm')}
             style={styles.plotImg}
        />
      </Paper>
    </Grid>

    <Grid item xs={12}>
      <Paper>
        <h4>True Positives</h4>
        <img
            src={
              plotUrl(selectedPlot, 'true_positives')}
            style={styles.plotImg}
        />
      </Paper>
    </Grid>

    <Grid item xs={12}>
      <Paper>
        <h4>False Positives</h4>
        <img src={
          plotUrl(selectedPlot, 'false_positives')}
             style={styles.plotImg}
        />
      </Paper>
    </Grid>


    <Grid item xs={12}>
      <h4>True Negatives</h4>
      <Paper>
        <img src={
          plotUrl(selectedPlot, 'true_negatives')}
             style={styles.plotImg}
        />
      </Paper>
    </Grid>


    <Grid item xs={12}>
      <Paper>
        <h4>False Negatives</h4>
        <img src={
          plotUrl(selectedPlot, 'false_negatives')}
             style={styles.plotImg}
        />
      </Paper>
    </Grid>
  </Grid>);
};

export default TrainerResultsView;