import React from 'react';
import Grid from '@material-ui/core/Grid';

const TrainerDataView = ({ dataName, imageNames, offset, parentStyles }) => {
  // TODO: Labels with the images
  const baseURL = 'https://storage.googleapis.com/elvos-public/processed';

  // Render 15 of the items
  const images = imageNames
      .slice(offset, offset + 15)
      .map((name) => {
        return (
            <Grid item xs={4}>
              <img src={`${baseURL}/${dataName}/arrays/${name}`}/>
            </Grid>);
      });

  return (
      <Grid container spacing={8} style={parentStyles.grid}>
        <Grid item xs={12}><h2>Dataset: {dataName}</h2></Grid>
        {images}
      </Grid>
  );
};

export default TrainerDataView;
