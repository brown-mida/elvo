const merge = require('webpack-merge');
const common = require('./webpack.config.js');

console.log(merge(common, {
  mode: 'production',
}));