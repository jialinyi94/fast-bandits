import numpy


def update_mean_and_counts(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    arms: numpy.ndarray,
    rewards: numpy.ndarray,
):
    """Update the mean rewards and trial counts for each arm.

    Parameters
    ----------
    mean_rewards : numpy.ndarray
        a numpy array of shape (..., num_arms) where the mean reward for each arm is stored.
    trial_counts : numpy.ndarray
        a numpy array of shape (..., num_arms) where the number of trials for each arm is stored.
    arms : numpy.ndarray
        a numpy array of shape (..., num_rounds) where the selected arms are stored.
    rewards : numpy.ndarray
        a numpy array of shape (..., num_rounds) where the rewards for the selected arms are stored.
    
    Returns
    -------
    new_means : numpy.ndarray
        a numpy array of shape (..., num_arms) where the updated mean reward for each arm is stored.
    new_counts : numpy.ndarray
        a numpy array of shape (..., num_arms) where the updated number of trials for each arm is stored.
    """
    *_, num_arms = mean_rewards.shape
    embedding = (arms[..., numpy.newaxis] == numpy.arange(num_arms)).astype(int)
    inc_rewards = numpy.sum(rewards[..., numpy.newaxis] * embedding, axis=-2)
    inc_counts = numpy.sum(embedding, axis=-2)
    old_sum = mean_rewards * trial_counts
    new_sum = old_sum + inc_rewards
    new_counts = trial_counts + inc_counts
    new_means = new_sum / new_counts
    return new_means, new_counts
