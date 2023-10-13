##############################################################
# Convolution in 2D example
# =========================


##############################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.

import numpy as np
import torch
import pytorch_finufft
import matplotlib.pyplot as plt

##############################################################
# Let's create a Gaussian convolutional filter as a function of x,y

def gaussian_function(x, y, sigma=1):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


##############################################################
# Let's visualize this filter kernel. We will be using it to convolve with points living on
# the [0, 2*pi] x [0, 2*pi] torus. So let's dimension it accordingly.

shape = (128, 128)
sigma = 0.5
x = np.linspace(-np.pi, np.pi, shape[0], endpoint=False)
y = np.linspace(-np.pi, np.pi, shape[1], endpoint=False)

gaussian_kernel = gaussian_function(x[:, np.newaxis], y, sigma=sigma)

fig, ax = plt.subplots()
ax.imshow(gaussian_kernel)

##############################################################
# In order for the kernel to not shift the signal, we need to place its mass at 0
# To do this, we ifftshift the kernel

shifted_gaussian_kernel = np.fft.ifftshift(gaussian_kernel)

fig, ax = plt.subplots()
ax.imshow(shifted_gaussian_kernel)


##############################################################
# Now let's create a point cloud on the torus that we can convolve with our filter

N = 20
points = np.random.rand(2, N) * 2 * np.pi

fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_aspect('equal')
ax.scatter(points[0], points[1], s=1)

##############################################################
# Now we can convolve the point cloud with the filter kernel.
# To do this, we Fourier-transform both the point cloud and the filter kernel,
# multiply them together, and then inverse Fourier-transform the result.
# First we need to convert all data to torch tensors

fourier_shifted_gaussian_kernel = torch.fft.fft2(torch.from_numpy(shifted_gaussian_kernel))
fourier_points = pytorch_finufft.functional.finufft_type1.apply(torch.from_numpy(points), torch.ones(points.shape[1], dtype=torch.complex128), shape)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(fourier_shifted_gaussian_kernel.real)
axs[1].imshow(fourier_points.real, vmin=-10, vmax=10)
axs[2].imshow((fourier_points * fourier_shifted_gaussian_kernel / fourier_shifted_gaussian_kernel[0, 0]).real, vmin=-10, vmax=10)


##############################################################
# We now have two possibilities: Invert the Fourier transform on a grid, or on a point cloud.
# We'll first invert the Fourier transform on a grid in order to be able to visualize the effect of the convolution.

convolved_points = torch.fft.ifft2(fourier_points * fourier_shifted_gaussian_kernel)

fig, ax = plt.subplots()
ax.imshow(convolved_points.real)
ax.scatter(points[1] / 2 / np.pi * shape[0], points[0] / 2 / np.pi * shape[1], s=2, c='r')

##############################################################
# We see that the convolution has smeared out the point cloud.
# After a small coordinate change, we can also plot the original points
# on the same plot as the convolved points.


##############################################################
# Next, we invert the Fourier transform on the same points as 
# our original point cloud. We will then compare this to direct evaluation
# of the kernel on all pairwise difference vectors between the points.

convolved_at_points = pytorch_finufft.functional.finufft_type2.apply(
    torch.from_numpy(points), fourier_points * fourier_shifted_gaussian_kernel,
    None, {'isign': 1}
    ).real / np.prod(shape)

fig, ax = plt.subplots()
ax.imshow(convolved_points.real)
ax.scatter(points[1] / 2 / np.pi * shape[0], points[0] / 2 / np.pi * shape[1], s=10 * convolved_at_points, c='r')

##############################################################
# To compute the convolution directly, we need to evaluate the kernel on all pairwise difference vectors between the points.

pairwise_diffs = points[:, np.newaxis] - points[:, :, np.newaxis]
kernel_diff_evals = gaussian_function(*pairwise_diffs, sigma=sigma)
convolved_by_hand = kernel_diff_evals.sum(1)

fig, ax = plt.subplots()
ax.plot(convolved_at_points.numpy(), convolved_by_hand, ".")
ax.plot([1, 3], [1, 3])

print(f"Relative difference between fourier convolution and direct convolution {torch.norm(convolved_at_points - convolved_by_hand) / np.linalg.norm(convolved_by_hand)}")
