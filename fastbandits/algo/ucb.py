import numpy

from fastbandits.core.regret import realized_rewards
from fastbandits.core.statistics import update_mean_and_counts


def score(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    t: int,
):
    """Compute the UCB score for each arm.

    Parameters
    ----------
    mean_rewards : numpy.ndarray
        a numpy array of shape (..., num_arms) where the mean rewards for each arm are stored.
    trial_counts : numpy.ndarray
        a numpy array of shape (..., num_arms) where the number of trials for each arm are stored.
    t : int
        the current time step.

    Returns
    -------
    scores : numpy.ndarray
        a numpy array of shape (..., num_arms) where the UCB score for each arm is stored.
    """
    exploration_bonus = numpy.sqrt(2 * numpy.log(t) / trial_counts)
    return mean_rewards + exploration_bonus


def select_arm(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    t: int,
):
    """Select the arm with the highest UCB score.

    Parameters
    ----------
    mean_rewards : numpy.ndarray
        a numpy array of shape (..., num_arms) where the mean rewards for each arm are stored.
    trial_counts : numpy.ndarray
        a numpy array of shape (..., num_arms) where the number of trials for each arm are stored.
    t : int
        the current time step.

    Returns
    -------
    numpy.ndarray
        a numpy array of shape (...) where the index of the selected arm is stored.
    """
    ucb_scores = score(mean_rewards, trial_counts, t)
    return numpy.argmax(ucb_scores, axis=-1)


def update(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    arm: int,
    reward: float,
):
    pass


def play(
    env: numpy.ndarray,
    prior_mean_rewards: numpy.ndarray | None,
    prior_trial_counts: numpy.ndarray | None,
):
    pass