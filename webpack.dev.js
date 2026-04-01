const path = require('path');
const { merge } = require('webpack-merge');
const common = require('./webpack.common.js');

module.exports = merge(common, {
    mode: 'development',
    devtool: 'inline-source-map',
    output: {
        filename: '[name].bundle.js',
        path: path.resolve(__dirname, 'dist'),
        clean: true
    },
    devServer: {
        static: path.join(__dirname, "dist"),
        compress: true,
        port: 3003,
        proxy: [
            {
                context: ['/api'],
                target: 'http://localhost:3007',
                changeOrigin: true,
            },
        ],
    }
})
