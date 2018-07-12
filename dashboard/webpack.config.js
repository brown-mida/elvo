const webpack = require('webpack');
module.exports = {
  mode: 'development',
  entry: [
    "./js/index.js"
  ],
  output: {
    path: __dirname + '/static',
    filename: "bundle.js"
  },
  module: {
    rules: [
      {
        test: /\.js?$/,
        loader: 'babel-loader',
        query: {
          presets: ['es2015', 'react']
        },
        exclude: /node_modules/
      }
    ]
  },
  plugins: []
};

