import numpy

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

    ::math::
        UCB = mean_rewards + sqrt(2 * log(t) / trial_counts if trial_counts > 0 else inf
    """
    exploration_bonus = numpy.inf * numpy.ones_like(mean_rewards)
    warmup = trial_counts > 0
    if t > 0:
        exploration_bonus[warmup] = numpy.sqrt(2 * numpy.log(t) / trial_counts[warmup])
    ucb = mean_rewards + exploration_bonus
    return ucb


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
    arms: numpy.ndarray,
    rewards: numpy.ndarray,
):
    """Update the states, i.e. mean_reward and trial_count, for the selected arms.

    Parameters
    ----------
    mean_rewards : numpy.ndarray
        a numpy array of shape (..., num_arms) where the mean rewards for each arm are stored.
    trial_counts : numpy.ndarray
        a numpy array of shape (..., num_arms) where the number of trials for each arm are stored.
    arms : numpy.ndarray
        a numpy array of shape (...) where the index of the selected arm is stored.
    rewards : numpy.ndarray
        a numpy array of shape (...) where the reward for the selected arm is stored.

    Returns
    -------
    mean_rewards : numpy.ndarray
        a numpy array of shape (..., num_arms) where the updated mean rewards for each arm are stored.
    trial_counts : numpy.ndarray
        a numpy array of shape (..., num_arms) where the updated number of trials for each arm are stored.
    """
    return update_mean_and_counts(mean_rewards, trial_counts, arms, rewards)


def rollout(
    env: numpy.ndarray,
    prior_mean_rewards: numpy.ndarray | None,
    prior_trial_counts: numpy.ndarray | None,
):
    """Rollout the UCB algorithm on the environment.
    """
    pass


if __name__ == "__main__":
    # test score function, when mean and trials have 3d
    mean_rewards = numpy.array([[0.5, 0.5], [0.5, 0.5]])
    trial_counts = numpy.array([[0, 0], [0, 0]])
    ucb_scores = score(mean_rewards, trial_counts, 0)
    print(ucb_scores)