"""
What type of tests should be here?
1) Tests for how matrix looks for delta_E & annihilation angle for:
    gt == True
    gt == False
    Meaning 2 tests in total.
    For gt == True, the concentration should be around delta_E = 0 & annihilation_angle = -1 (180 degrees).
    For gt == False, it should be more spread.

2) Tests for how matrix looks for delta_E & ARM for:
    gt == True
    gt == False
    Meaning 2 tests in total.
    For gt == True, the concentration should be around delta_E = 0 & ARM = 0.
    For gt == False, it should be more spread.

3) Tests for how good ARM truly works
    More specifically, how good the geometric angle is & how good the kinetic angle is

4) How good the annihilation_angle algorithm reconstructs angle
    Here we should go into the .sim file, find some examples of IA ANNI's, and look at their photon output vectors (that should be 180 degreees),
    and check how close we get to that in our reconstructions.

5) How much deviance there is from the annihilation algorithm reconstruction from the actual truth in average and have an assertion for that
   so that we can make sure that the deviance isn't too big.

Create plots for these tests!
"""

# Energy, etc tests
# TODO: Test nr 1 from the top-level docstring
# TODO: Test nr 2 from the top-level docstring

# ARM
# TODO: Test nr 3 from the top-level docstring

# Annihilation Angle
# TODO: Test nr 4 from the top-level docstring
# TODO: Test nr 5 from the top-level docstring
