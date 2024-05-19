import { resolve } from 'path';
import TerserPlugin from "terser-webpack-plugin";
import common from './webpack.common.js';

import CssMinimizerPlugin from "css-minimizer-webpack-plugin";
import { merge } from 'webpack-merge';

export default merge(common, {
    mode: 'production',
    devtool: 'source-map',
    output: {
        filename: '[name].[contenthash].bundle.js',
        path: resolve(__dirname, 'dist'),
        clean: true
    },
    optimization: {
        minimizer: [
            new CssMinimizerPlugin(),
            new TerserPlugin(),
        ],
    },
})
 