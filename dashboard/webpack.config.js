const webpack = require('webpack');
module.exports = {
  // TODO(luke): Separate production webpack build.
  mode: 'development',
  entry: {
    annotator: './js/annotator.js',
    trainer: './js/trainer.js',
  },
  output: {
    path: __dirname + '/static',
    filename: '[name].bundle.js',
  },
  module: {
    rules: [
      {
        test: /\.js?$/,
        loader: 'babel-loader',
        query: {
          presets: ['es2015', 'react'],
        },
        exclude: /node_modules/,
      },
    ],
  },
  plugins: [],
};

