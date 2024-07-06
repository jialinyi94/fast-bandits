import numpy
import fastbandits.algo.ucb as ucb

def test_score_nd():
    mean_rewards = numpy.array(
        [
            [0.1, 0.2, 0.3], 
            [0.4, 0.5, 0.6]
        ]
    )
    trial_counts = numpy.array(
        [
            [0, 10, 2], 
            [3, 4, 5]
        ]
    )
    t = 12
    scores = ucb.score(t, mean_rewards, trial_counts)
    assert numpy.allclose(
        scores, 
        [
            [numpy.inf, 0.2 + numpy.sqrt(2 * numpy.log(t) / 10), 0.3 + numpy.sqrt(2 * numpy.log(t) / 2)], 
            [0.4 + numpy.sqrt(2 * numpy.log(t) / 3), 0.5 + numpy.sqrt(2 * numpy.log(t) / 4), 0.6 + numpy.sqrt(2 * numpy.log(t) / 5)]
        ]
    )


def test_select_arm_nd():
    mean_rewards = numpy.array(
        [
            [0.1, 0.2, 0.3], 
            [0.4, 0.5, 0.6]
        ]
    )
    trial_counts = numpy.array(
        [
            [0, 10, 2], 
            [3, 4, 5]
        ]
    )
    t = 12
    selected_arm = ucb.select_arm(t, mean_rewards, trial_counts)
    assert numpy.allclose(selected_arm, [0, 0])