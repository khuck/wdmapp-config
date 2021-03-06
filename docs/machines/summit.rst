
WDMApp on Summit
*****************************

Setting up Spack
====================

Follow the generic instructions from  :ref:`setup-spack-label` to install Spack and add the
WDMapp Spack package repository.

Summit-Specific Setup
-------------------------

You can copy your choice of a basic or a more comprehensive setup for
Spack on Summit from the
`<https://github.com/wdmapp/wdmapp-config/summit/spack>`_ repository.

.. code-block:: sh

  $ mkdir -p ~/.spack/linux		
  $ cp path/to/wdmapp-config/summit/spack/*.yaml ~/.spack/linux/

.. warning::

   This will overwrite an existing Spack configuration, so be careful
   if you've previously set Spack up. If you have an existing config, consider
   using ``path/to/spack/etc/spack/package.yaml`` for packages instead, and add
   gcc 8.1.1 to your exising ``compilers.yaml`` if not already present.

If you use the provided ``packages.yaml``, it only tells Spack about
essential existing pre-installed packages on Summit, ie., CUDA, MPI
and the corresponding compilers. Spack will therefore build and
install all other dependencies from scratch, which takes time but has
the advantage that it'll generate pretty much the same software stack
on any machine you use.

On the other hand, ``packages-extended.yaml`` (which needs to be
renamed to ``packages.yaml`` to use it), tells Spack comprehensively
about pre-installed software on Summit, so installation of WDMapp will
proceed more quickly and use system-provided libraries where possible.

.. note::

   Make sure that you don't have a ``spectrum-mpi`` loaded. By
   default, Summit will load the ``xl`` and ``spectrum-mpi`` modules
   for you, and those interfere when Spack tries to perform ``gcc``
   based builds.  You might want to consider adding this to your
   ``.bashrc`` or similar init file:

   .. code-block:: shell

      module unload xl spectrum-mpi
   
.. note::

   On Summit, the cuda module sets environment variables that set a
   path which ``nvcc`` does not otherwise add. Because of this, it is
   requried to ``module load cuda/10.1.243`` before building GENE, and
   probably other software that uses CUDA..

Consider also configuring spack to use gpfs scratch space (i.e. ``$MEMBERWORK``)
when building packages, rather than the home filesystem which tends to have
problems with high workload tasks:

.. code-block:: sh

  $ mkdir -p /gpfs/alpine/scratch/$USER/spack-stage

and add the following to ``~/.spack/config.yaml``:

.. code-block:: yaml

  config:
    build_stage: /gpfs/alpine/scratch/$user/spack-stage
   


Building WDMapp
================

You should be able to just follow the generic instructions from
:ref:`build-wdmapp-label`.

Running a Sample Job
====================

.. todo::

   Complete instructions on how to get the test case set up and run.

You can get the setup for a coupled WDMapp run by cloning
https://github.com/wdmapp/testcases.

The sample sample job from
https://github.com/wdmapp/wdmapp-config/longhorn/submit_wdmapp.sh will
run the `run_1` coupled case.

.. literalinclude:: ../../summit/submit_wdmapp.sh
   :language: shell
   :linenos:

Submit as usal:

.. code-block:: sh

   $ bsub submit_wdmapp.sh

   
