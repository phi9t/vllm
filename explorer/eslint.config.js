import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
    rules: {
      // Honor the `_`-prefix convention for intentionally-unused bindings
      // (e.g. dropping react-markdown's `node` prop, unused render args).
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      // shadcn-style ui files export a CVA `*Variants` const beside the
      // component; allow it for Fast Refresh.
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
      // The mode views intentionally reset stale state when the selected
      // subject changes, then fetch — a benign, idiomatic data-reset effect.
      'react-hooks/set-state-in-effect': 'warn',
    },
  },
])
