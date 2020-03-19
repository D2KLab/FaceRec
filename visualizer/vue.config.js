const nib = require('nib');
const rupture = require('rupture');

module.exports = {
  runtimeCompiler: true,
  css: {
    loaderOptions: {
      stylus: {
        use: [nib(), rupture()],
      },
    },
  },
};
