.. image:: https://img.shields.io/pypi/v/dwave-optimization.svg
    :target: https://pypi.org/project/dwave-optimization

.. image:: https://img.shields.io/pypi/pyversions/dwave-optimization.svg
    :target: https://pypi.python.org/pypi/dwave-optimization

.. image:: https://circleci.com/gh/dwavesystems/dwave-optimization.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-optimization

dwave-optimization
==================

.. index-start-marker1

`dwave-optimization` enables the formulation of nonlinear models for 
industrial optimization problems. The package includes:

*   a class for nonlinear models used by the 
    `Leap <https://cloud.dwavesys.com/leap>`_ service's 
    quantum-classical hybrid nonlinear-program solver.
*   model generators for common optimization problems.

.. index-end-marker1

(For explanations of the terminology, see the
`Ocean glossary <https://docs.ocean.dwavesys.com/en/stable/concepts/index.html>`_.)

Example Usage
-------------

.. index-start-marker2

The  
`flow-shop scheduling <https://en.wikipedia.org/wiki/Flow-shop_scheduling>`_ 
problem is a variant of the renowned 
`job-shop scheduling <https://en.wikipedia.org/wiki/Optimal_job_scheduling>`_ 
optimization problem. Given ``n`` jobs to schedule on ``m`` machines, with 
specified processing times for each job per machine, minimize the makespan 
(the total length of the schedule for processing all the jobs). For every 
job, the ``i``-th operation is executed on the ``i``-th machine. No machine 
can perform more than one operation simultaneously. 

This small example builds a model for optimizing the schedule for processing 
two jobs on three machines.

.. code-block:: python

    from dwave.optimization.generators import flow_shop_scheduling
    
    processing_times = [[10, 5, 7], [20, 10, 15]]
    model = flow_shop_scheduling(processing_times=processing_times)

.. index-end-marker2

See the `documentation <https://docs.ocean.dwavesys.com/en/stable/docs_optimization/>`_
for more examples.

Installation
------------

.. installation-start-marker

Installation from `PyPI <https://pypi.org/project/dwave-optimization>`_:

.. code-block:: bash

    pip install dwave-optimization

During package development, it is often convenient to use an editable install.
See `meson-python's editible installs 
<https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html>`_
for more details.

.. code-block:: bash

    pip install -r requirements.txt
    pip install --no-build-isolation --config-settings=editable-verbose=true --editable .

.. installation-end-marker

Testing
-------

All code should be thoroughly tested and all pull requests should include tests.

To run the Python tests, first install the package using an editable install
as described above. The tests can then be run with
`unittest <https://docs.python.org/3/library/unittest.html>`_.

.. code-block:: bash

    python -m unittest

To run the C++ tests, first install the project dependencies, then setup a
``meson`` build directory. You must configure the build as a debug build for
the tests to run.

.. code-block:: bash

    pip install -r requirements.txt
    meson setup build -Dbuildtype=debug

You can then run the tests using
`meson's test framework <https://mesonbuild.com/Unit-tests.html>`_.

.. code-block:: bash

    meson test -Cbuild

License
-------

Released under the Apache License 2.0. See LICENSE file.

Contributing
------------

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_
has guidelines for contributing to Ocean packages.

``dwave-optimization`` includes some formatting customization in the
`.clang-format <.clang-format>`_ and `setup.cfg <setup.cfg>`_ files.

Release Notes
~~~~~~~~~~~~~

``dwave-optimization`` makes use of `reno <https://docs.openstack.org/reno/>`_
to manage its release notes.

When making a contribution to ``dwave-optimization`` that will affect users,
create a new release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
