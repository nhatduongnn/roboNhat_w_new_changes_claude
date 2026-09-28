"""calculateKD without importing robocop.utils.parameterize.

A verbatim copy of pkg/robocop/utils/parameterize.py `calculateKD` (same operations in the same
order, so results are bit-identical). It exists because importing `parameterize` also imports
`robocop.utils.parameters`, whose module-level rpy2 import starts an embedded R. The tuner's
trainDir builds call calculateKD every round, and R's startup intermittently fails with
`no item called "package:utils" on the search list` -- which killed tuning rounds on 2026-09-16,
-17, -18, -19 and -20. Nothing in calculateKD uses R.

Moving the import inside a function (the 2026-09-20 attempt) did not help: the function runs on
every build, so R still started, just later. This module removes the dependency instead.
Bit-identity against parameterize.calculateKD is checked for every motif in every source trainDir
the tuner uses (see the 2026-09-21 chain.log resume notes).
"""
import math

import numpy as np


def calculateKD(pwm, k):
    score = 0
    for i in range(len(pwm[k][0])):
        idx = np.argmax(pwm[k][:, i])
        score += math.log10(np.ravel(pwm['background'])[idx]) - math.log10(pwm[k][idx, i])
    return 10**score
