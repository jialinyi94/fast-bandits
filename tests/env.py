import numpy 
import fastbandits.algo.ucb as ucb

def test_score_1d():
    mean_rewards = numpy.array([0.1, 0.2, 0.3])
    trial_counts = numpy.array([0, 1, 2])
    t = 3
    scores = ucb.score(t, mean_rewards, trial_counts)
    assert numpy.allclose(scores, [numpy.inf, 0.2 + numpy.sqrt(2 * numpy.log(t) / 2), 0.3 + numpy.sqrt(2 * numpy.log(t) / 2)])

def test_select_arm_1d():
    mean_rewards = numpy.array([0.1, 0.2, 0.3])
    trial_counts = numpy.array([0, 1, 2])
    t = 3
    selected_arm = ucb.select_arm(t, mean_rewards, trial_counts)
    assert selected_arm == 0