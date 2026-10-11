const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const CopyPlugin = require("copy-webpack-plugin");
const path = require('path');
const fs = require('fs');

const neuralModelContract = require('./src/game/controllers/neural-model-contract.json');

function contractMatchesRuntime(contractPath) {
    try {
        const candidate = JSON.parse(fs.readFileSync(contractPath, 'utf8'));
        return candidate.contractVersion === neuralModelContract.contractVersion
            && candidate.observationVersion === neuralModelContract.observationVersion
            && candidate.actionVersion === neuralModelContract.actionVersion
            && JSON.stringify(candidate.observation) === JSON.stringify(neuralModelContract.observation)
            && JSON.stringify(candidate.action) === JSON.stringify(neuralModelContract.action);
    } catch {
        return false;
    }
}

function isCompatibleOnnx(resourcePath) {
    return contractMatchesRuntime(resourcePath.replace(/\.onnx$/i, '.contract.json'))
        && fs.existsSync(resourcePath.replace(/\.onnx$/i, '.weights.json'));
}

function isCompatibleContract(resourcePath) {
    return contractMatchesRuntime(resourcePath)
        && fs.existsSync(resourcePath.replace(/\.contract\.json$/i, '.weights.json'));
}

module.exports = {
    entry: './src/index.ts',
    plugins: [
        new HtmlWebpackPlugin({
            title: 'Treads',
            template: './src/index.html',
            favicon: './src/assets/icons/favicon.ico',
            scriptLoading: 'module',
            meta: {
                description: 'Treads',
                charset: 'UTF-8',
                viewport: 'width=device-width, initial-scale=1, shrink-to-fit=no'
            }
        }),
        new MiniCssExtractPlugin({
            filename: "[name].css",
        }),
        new CopyPlugin({
            patterns: [
                {
                    from: 'training/output/*.onnx',
                    to: 'models/[name][ext]',
                    noErrorOnMissing: true,
                    filter: isCompatibleOnnx,
                },
                {
                    from: 'training/output/*.contract.json',
                    to: 'models/[name][ext]',
                    noErrorOnMissing: true,
                    filter: isCompatibleContract,
                },
                {
                    from: 'training/output/*.weights.json',
                    to: 'models/[name][ext]',
                    noErrorOnMissing: true,
                    filter: (resourcePath) => contractMatchesRuntime(resourcePath.replace(/\.weights\.json$/i, '.contract.json')),
                },
            ],
        }),
    ],
    optimization: {
        moduleIds: 'deterministic',
        runtimeChunk: 'single',
        splitChunks: {
            chunks: 'all',
            cacheGroups: {
                vendor: {
                    test: /[\\/]node_modules[\\/]/,
                    name: 'vendors',
                    chunks: 'all',
                },
            },
        },
    },
    performance: {
        assetFilter: (assetFilename) => !/\.(mp3|ogg|wav|onnx|map)$|\.(weights|contract)\.json$/i.test(assetFilename),
    },
    module: {
        rules: [
            {
                test: /\.css$/,
                use: [MiniCssExtractPlugin.loader, "css-loader"]
            },
            {
                test: /\.(png|svg|jpg|jpeg|gif|ogg|mp3|wav)$/i,
                type: 'asset/resource',
            },
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: /node_modules/,
            },
            {
                test: /\.ico$/,
                use: [
                    {
                        loader: 'file-loader',
                        options: {
                            name: '[name].ico',
                            outputPath: 'assets/icons',
                        },
                    },
                ],
            },
        ],
    },
    resolve: {
        extensions: ['.js', '.jsx', '.ts', '.tsx', '...'],
    },
}
