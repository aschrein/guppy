const path = require('path')

module.exports = function override(config, env) {
  {
    const wasmExtensionRegExp = /\.wasm$/;

    config.resolve.extensions.push('.wasm');

    config.module.rules.forEach(rule => {
      (rule.oneOf || []).forEach(oneOf => {
        if (oneOf.loader && oneOf.loader.indexOf('file-loader') >= 0) {
          // Make file-loader ignore WASM files
          oneOf.exclude.push(wasmExtensionRegExp);
        }
      });
    });

    // Add a dedicated loader for WASM
    config.module.rules.push({
      test: wasmExtensionRegExp,
      include: path.resolve(__dirname, 'src'),
      use: [{ loader: require.resolve('wasm-loader'), options: {} }]
    });
  }
  {
    const RegExp = /\.s$|\.md/;

    config.resolve.extensions.push('.s');
    config.resolve.extensions.push('.md');

    config.module.rules.forEach(rule => {
      (rule.oneOf || []).forEach(oneOf => {
        if (oneOf.loader && oneOf.loader.indexOf('file-loader') >= 0) {
          oneOf.exclude.push(RegExp);
        }
      });
    });

    config.module.rules.push({
      test: RegExp,
      include: path.resolve(__dirname, 'src'),
      use: [{ loader: require.resolve('raw-loader'), options: {} }]
    });
  }
  return config;
}