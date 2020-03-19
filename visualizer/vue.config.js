const nib = require('nib');
const rupture = require('rupture');

module.exports = {
  runtimeCompiler: true,
  publicPath: '/visualizer',
  css: {
    loaderOptions: {
      stylus: {
        use: [nib(), rupture()],
      },
    },
  },
};
