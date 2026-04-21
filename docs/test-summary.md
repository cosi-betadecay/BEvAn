# Test Summary

<!-- This file is generated from a pytest JUnit XML report. -->

- Source report: `artifacts/pytest-junit.xml`
- Environment: local ROOT-enabled environment
- Last run: 2026-04-12T05:21:43.443451+02:00

## Overall

- Total: 42
- Passed: 41
- Failed: 1
- Errors: 0
- Skipped: 0
- Duration: 559.196s

## Test Suites

- `pytest`: 41 passed, 1 failed, 0 errors, 0 skipped in 559.196s

## Failures And Errors

- `tests.tests_physics.test_matrix_calculations::test_concentrated_data_fills_single_cell` (failure)
  AssertionError: Expected max cell probability > 0.9, got 0.1827 assert 0.18274112045764923 > 0.9 + where 0.18274112045764923 = <built-in method item of Tensor object at 0xe47a61bde520>() + where <built-in method item...

## Notes

- These results come from a local ROOT-enabled environment.
- Regenerate this file after running pytest to keep the documentation current.

