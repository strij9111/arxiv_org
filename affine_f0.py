"""
In this example, we first define the functions compute_target_mean_std() and affine_transform_f0().
Then, we use them to compute the target mean and standard deviation of the f0 trajectories for male and female
speakers, and perform the affine transformation on a sample f0 trajectory.

When you run this code, it should output the target mean and standard deviation, as well as the original and
transformed f0 trajectories.
"""
import numpy as np


def compute_target_mean_std(male_f0s, female_f0s):
    male_mean = np.mean(male_f0s)
    male_std = np.std(male_f0s)
    female_mean = np.mean(female_f0s)
    female_std = np.std(female_f0s)

    target_mean = (male_mean + female_mean) / 2
    target_std = (male_std + female_std) / 2

    return target_mean, target_std


def affine_transform_f0(f0_trajectory, target_mean, target_std):
    source_mean = np.mean(f0_trajectory)
    source_std = np.std(f0_trajectory)

    transformed_f0 = (f0_trajectory - source_mean) * (target_std / source_std) + target_mean
    return transformed_f0


# Example usage
male_f0s = np.array([100, 110, 120, 130, 140])
female_f0s = np.array([180, 190, 200, 210, 220])

target_mean, target_std = compute_target_mean_std(male_f0s, female_f0s)
print("Target mean:", target_mean)
print("Target std:", target_std)

f0_trajectory = np.array([100, 120, 130, 110, 140])

transformed_f0 = affine_transform_f0(f0_trajectory, target_mean, target_std)
print("Original f0 trajectory:", f0_trajectory)
print("Transformed f0 trajectory:", transformed_f0)
