const eslint = require('@eslint/js');
const tseslint = require('typescript-eslint');

module.exports = tseslint.config(
	eslint.configs.recommended,
	...tseslint.configs.strict,
	...tseslint.configs.stylistic,
	{
		ignores: ['dist/**/*.*', '.test-dist/**/*.*', '.training-dist/**/*.*', 'webpack.*.js', 'eslint.config.*', 'training/test-cli-runner.js'],
	},
	{
		rules: {
			'@typescript-eslint/no-unused-vars': [
				'error',
				{
					argsIgnorePattern: '^_',
					varsIgnorePattern: '^_',
				},
			],
		},
	}
);
