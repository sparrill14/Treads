import HtmlWebpackPlugin from 'html-webpack-plugin';
import MiniCssExtractPlugin, { loader as _loader } from "mini-css-extract-plugin";

export const entry = './src/index.ts';
export const plugins = [
    new HtmlWebpackPlugin({
        title: 'Treads',
        template: './src/index.html'
    }),
    new MiniCssExtractPlugin({
        filename: "[name].css",
    }),
];
export const optimization = {
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
};
export const module = {
    rules: [
        {
            test: /\.css$/,
            use: [_loader, "css-loader"]
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
                        outputPath: 'assets',
                    },
                },
            ],
        },
    ],
};
export const resolve = {
    extensions: ['.js', '.jsx', '.ts', '.tsx', '...'],
};