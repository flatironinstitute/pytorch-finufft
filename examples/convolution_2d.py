"""
Convolution in 2D
=================
"""

#######################################################################################
# Import packages
# ---------------
#
# First, we import the packages we need for this example.

import matplotlib.pyplot as plt
import numpy as np
import torch

import pytorch_finufft

#######################################################################################
# Let's create a Gaussian convolutional filter as a function of x,y


def gaussian_function(x, y, sigma=1):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))


#######################################################################################
# Let's visualize this filter kernel. We will be using it to convolve with points
# living on the $[0, 2*\pi] \times [0, 2*\pi]$ torus. So let's dimension it accordingly.

shape = (128, 128)
sigma = 0.5
x = np.linspace(-np.pi, np.pi, shape[0], endpoint=False)
y = np.linspace(-np.pi, np.pi, shape[1], endpoint=False)

gaussian_kernel = gaussian_function(x[:, np.newaxis], y, sigma=sigma)

fig, ax = plt.subplots()
_ = ax.imshow(gaussian_kernel)

#######################################################################################
# In order for the kernel to not shift the signal, we need to place its mass at 0.
# To do this, we ifftshift the kernel

shifted_gaussian_kernel = np.fft.ifftshift(gaussian_kernel)

fig, ax = plt.subplots()
_ = ax.imshow(shifted_gaussian_kernel)


#######################################################################################
# Now let's create a point cloud on the torus that we can convolve with our filter

N = 20
points = np.random.rand(2, N) * 2 * np.pi

fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(0, 2 * np.pi)
ax.set_aspect("equal")
_ = ax.scatter(points[0], points[1], s=1)


#######################################################################################
# Now we can convolve the point cloud with the filter kernel.
# To do this, we Fourier-transform both the point cloud and the filter kernel,
# multiply them together, and then inverse Fourier-transform the result.
# First we need to convert all data to torch tensors

fourier_shifted_gaussian_kernel = torch.fft.fft2(
    torch.from_numpy(shifted_gaussian_kernel)
)
fourier_points = pytorch_finufft.functional.finufft_type1(
    torch.from_numpy(points), torch.ones(points.shape[1], dtype=torch.complex128), shape
)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(fourier_shifted_gaussian_kernel.real)
axs[1].imshow(fourier_points.real, vmin=-10, vmax=10)
_ = axs[2].imshow(
    (
        fourier_points
        * fourier_shifted_gaussian_kernel
        / fourier_shifted_gaussian_kernel[0, 0]
    ).real,
    vmin=-10,
    vmax=10,
)

#######################################################################################
# We now have two possibilities: Invert the Fourier transform on a grid, or on a point
# cloud. We'll first invert the Fourier transform on a grid in order to be able to
# visualize the effect of the convolution.

convolved_points = torch.fft.ifft2(fourier_points * fourier_shifted_gaussian_kernel)

fig, ax = plt.subplots()
ax.imshow(convolved_points.real)
_ = ax.scatter(
    points[1] / 2 / np.pi * shape[0], points[0] / 2 / np.pi * shape[1], s=2, c="r"
)

#######################################################################################
# We see that the convolution has smeared out the point cloud.
# After a small coordinate change, we can also plot the original points
# on the same plot as the convolved points.


#######################################################################################
# Next, we invert the Fourier transform on the same points as
# our original point cloud. We will then compare this to direct evaluation
# of the kernel on all pairwise difference vectors between the points.

convolved_at_points = pytorch_finufft.functional.finufft_type2(
    torch.from_numpy(points),
    fourier_points * fourier_shifted_gaussian_kernel,
    isign=1,
).real / np.prod(shape)

fig, ax = plt.subplots()
ax.imshow(convolved_points.real)
_ = ax.scatter(
    points[1] / 2 / np.pi * shape[0],
    points[0] / 2 / np.pi * shape[1],
    s=10 * convolved_at_points,
    c="r",
)

#######################################################################################
# To compute the convolution directly, we need to evaluate the kernel on all pairwise
# difference vectors between the points. Note the points that will be off the diagonal.
# These will be due to the periodic boundary conditions of the convolution.

pairwise_diffs = points[:, np.newaxis] - points[:, :, np.newaxis]
kernel_diff_evals = gaussian_function(*pairwise_diffs, sigma=sigma)
convolved_by_hand = kernel_diff_evals.sum(1)

fig, ax = plt.subplots()
ax.plot(convolved_at_points.numpy(), convolved_by_hand, ".")
ax.plot([1, 3], [1, 3])

relative_difference = torch.norm(
    convolved_at_points - convolved_by_hand
) / np.linalg.norm(convolved_by_hand)
print(
    "Relative difference between fourier convolution and direct convolution "
    f"{relative_difference}"
)


#######################################################################################
# Now let's see if we can learn the convolution kernel from the input and output point
# clouds. To this end, let's first make a pytorch object that can compute a kernel
# convolution on a point cloud.


class FourierPointConvolution(torch.nn.Module):
    def __init__(self, fourier_kernel_shape):
        super().__init__()
        self.fourier_kernel_shape = fourier_kernel_shape

        self.build()

    def build(self):
        self.register_parameter(
            "fourier_kernel",
            torch.nn.Parameter(
                torch.randn(self.fourier_kernel_shape, dtype=torch.complex128)
            ),
        )
        # ^ think about whether we need to scale this init in some better way

    def forward(self, points, values):
        fourier_transformed_input = pytorch_finufft.functional.finufft_type1(
            points, values, self.fourier_kernel_shape
        )
        fourier_convolved = fourier_transformed_input * self.fourier_kernel
        convolved = pytorch_finufft.functional.finufft_type2(
            points,
            fourier_convolved,
            isign=1,
        ).real / np.prod(self.fourier_kernel_shape)
        return convolved


#######################################################################################
# Now we can use this object in a pytorch training loop to learn the kernel from the
# input and output point clouds. We will use the mean squared error as a loss function.

fourier_point_convolution = FourierPointConvolution(shape)
optimizer = torch.optim.AdamW(
    fourier_point_convolution.parameters(), lr=0.005, weight_decay=0.001
)

ones = torch.ones(points.shape[1], dtype=torch.complex128)

losses = []
for i in range(10000):
    # Make new set of points and compute forward model
    points = np.random.rand(2, N) * 2 * np.pi
    torch_points = torch.from_numpy(points)
    fourier_points = pytorch_finufft.functional.finufft_type1(
        torch.from_numpy(points), ones, shape
    )
    convolved_at_points = pytorch_finufft.functional.finufft_type2(
        torch.from_numpy(points),
        fourier_points * fourier_shifted_gaussian_kernel,
        isign=1,
    ).real / np.prod(shape)

    # Learning step
    optimizer.zero_grad()
    convolved = fourier_point_convolution(torch_points, ones)
    loss = torch.nn.functional.mse_loss(convolved, convolved_at_points)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Iteration {i:05d}, Loss: {loss.item():1.4f}")


fig, ax = plt.subplots()
ax.plot(losses)
ax.set_ylabel("Loss")
ax.set_xlabel("Iteration")
ax.set_yscale("log")

fig, ax = plt.subplots()
im = ax.imshow(
    torch.real(torch.fft.fftshift(fourier_point_convolution.fourier_kernel.data))[
        48:80, 48:80
    ]
)
_ = fig.colorbar(im, ax=ax)
