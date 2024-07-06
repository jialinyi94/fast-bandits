import numpy

from fastbandits.core.regret import realized_rewards
from fastbandits.core.statistics import update_mean_and_counts


def _initialize(
    num_arms: int,
    prior_mean_rewards: numpy.ndarray,
    prior_trial_counts: numpy.ndarray,
):
    pass


def score(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    t: int,
):
    pass


def select_arm(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    t: int,
):
    pass


def update(
    mean_rewards: numpy.ndarray,
    trial_counts: numpy.ndarray,
    arm: int,
    reward: float,
):
    pass


def play(
    env: numpy.ndarray,
    prior_mean_rewards: numpy.ndarray,
    prior_trial_counts: numpy.ndarray,
):
    pass