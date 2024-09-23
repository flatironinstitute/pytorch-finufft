Installation
============


Pre-requistes
-------------

Pytorch-FINUFFT requires either ``finufft`` *and/or* ``cufinufft``
2.2.0 or greater.

These are available via `pip` or can be built from source.
See the relevant installation pages for
:external+finufft:doc:`finufft <install>` and
:external+finufft:doc:`cufinufft <install_gpu>`.


Pytorch-FINUFFT also requires ``pytorch`` 2.0 or greater. See the installation
matrix on `Pytorch's website <https://pytorch.org/get-started/>`_.


Source Installation
-------------------

Once the pre-requisites are installed, you can install Pytorch-FINUFFT
from source by running

.. code-block:: bash

    pip install -U git+https://github.com/flatironinstitute/pytorch-finufft.git


Installation from PyPI
----------------------

Pytorch-FINUFFT is available on PyPI and can be installed with

.. code-block:: bash

    pip install pytorch-finufft

which will also try to install compatible versions of torch and finufft.

You can also run

.. code-block:: bash

    pip install pytorch-finufft[cuda]

to additionally install cufinufft.

