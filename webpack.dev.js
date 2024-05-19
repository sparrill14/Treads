import { join, resolve } from 'path';
import { merge } from 'webpack-merge';
import common from './webpack.common.js';

export default merge(common, {
    mode: 'development',
    devtool: 'inline-source-map',
    output: {
        filename: '[name].bundle.js',
        path: resolve(__dirname, 'dist'),
        clean: true
    },
    devServer: {
        static: join(__dirname, "dist"),
        compress: true,
        port: 4000,
    }
})
 