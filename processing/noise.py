import numpy as np


# GAUSSIAN NOISE

def gaussian_noise(image, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# RAYLEIGH NOISE

def rayleigh_noise(image, scale=20):
    noise = np.random.rayleigh(scale, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

# GAMMA NOISE

def gamma_noise(image, shape=2.0, scale=10.0):
    noise = np.random.gamma(shape, scale, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# SALT & PEPPER NOISE

def salt_pepper_noise(image, prob=0.02):
    output = image.copy()
    rnd = np.random.random(image.shape)

    output[rnd < prob / 2] = 0
    output[rnd > 1 - prob / 2] = 255

    return output


# EXPONENTIAL NOISE

def exponential_noise(image, scale=20):
    noise = np.random.exponential(scale, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# UNIFORM NOISE

def uniform_noise(image, low=-20, high=20):
    noise = np.random.uniform(low, high, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
