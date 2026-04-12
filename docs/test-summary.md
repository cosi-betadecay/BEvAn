# Test Summary

<!-- This file is generated from a pytest JUnit XML report. -->

- Source report: `artifacts/pytest-junit.xml`
- Environment: local ROOT-enabled environment
- Last run: 2026-04-12T02:05:07.775692+02:00

## Overall

- Total: 37
- Passed: 32
- Failed: 5
- Errors: 0
- Skipped: 0
- Duration: 56.163s

## Test Suites

- `pytest`: 32 passed, 5 failed, 0 errors, 0 skipped in 56.163s

## Failures And Errors

- `tests.tests_physics.test_annihilation_detection_utils::test_true_bdecay_events_have_higher_beta_likelihood` (failure)
  AssertionError: True bdecay events: expected n_beta_arm sum (0.000000) > n_bg_arm sum (0.000000) assert tensor(0.) > tensor(0.)
- `tests.tests_physics.test_matrix_calculations::test_concentrated_data_fills_single_cell` (failure)
  AssertionError: Expected max cell probability > 0.9, got 0.1827 assert 0.18274112045764923 > 0.9 + where 0.18274112045764923 = <built-in method item of Tensor object at 0xf2cd2b6866b0>() + where <built-in method item...
- `tests.tests_physics.test_matrix_calculations::test_bg_arm_matrix_not_peaked_at_zero` (failure)
  AssertionError: Expected bdecay to have more mass near ARM=0 than background (bdecay=1.0000, bg=1.0000) assert 1.0 > 1.0
- `tests.tests_physics.test_matrix_calculations::test_majority_of_true_events_higher_bdecay_density_arm` (failure)
  AssertionError: Expected >55% of true bdecay events to have higher bdecay density (ARM pair), got nan% (n=0) assert nan > 0.55
- `tests.tests_physics.test_matrix_calculations::test_majority_of_false_events_higher_bg_density_arm` (failure)
  AssertionError: Expected >55% of background events to have higher bg density (ARM pair), got nan% (n=0) assert nan > 0.55

## Notes

- These results come from a local ROOT-enabled environment.
- Regenerate this file after running pytest to keep the documentation current.

