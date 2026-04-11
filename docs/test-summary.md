# Test Summary

<!-- This file is generated from a pytest JUnit XML report. -->

- Source report: `artifacts/pytest-junit.xml`
- Environment: local ROOT-enabled environment
- Last run: 2026-04-12T01:30:08.974653+02:00

## Overall

- Total: 23
- Passed: 21
- Failed: 2
- Errors: 0
- Skipped: 0
- Duration: 35.063s

## Test Suites

- `pytest`: 21 passed, 2 failed, 0 errors, 0 skipped in 35.063s

## Failures And Errors

- `tests.tests_physics.test_annihilation_detection_utils::test_true_bdecay_events_have_higher_beta_likelihood` (failure)
  AssertionError: True bdecay events: expected n_beta_arm sum (0.000000) > n_bg_arm sum (0.000000) assert tensor(0.) > tensor(0.)
- `tests.tests_physics.test_matrix_calculations::test_concentrated_data_fills_single_cell` (failure)
  AssertionError: Expected max cell probability > 0.9, got 0.1827 assert 0.18274112045764923 > 0.9 + where 0.18274112045764923 = <built-in method item of Tensor object at 0xfc06985f3e70>() + where <built-in method item...

## Notes

- These results come from a local ROOT-enabled environment.
- Regenerate this file after running pytest to keep the documentation current.

