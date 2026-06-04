============
Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages. With the following changes

* ``dwave-optimization`` uses C++20.
* ``dwave-optimization`` includes some formatting customization in the
  `.clang-format <.clang-format>`_ file.

Release Notes
=============

Pull request descriptions and commit messages are written for the developers of
the package, release notes are written for the end-user. These two audiences are
interested in different information.

Release notes tell the user about changes in a release that might affect them.
For example, new functions, bug fixes, and performance improvements all need
release notes.

Not every pull request needs a release note. Changes to docs, refactors without
a change in behavior, CI changes, package maintenace, etc., generally do not
require a release note. 

Creating a Release Note
-----------------------

``dwave-optimization`` makes use of `reno <https://docs.openstack.org/reno/>`_
to manage its release notes. Create a new release note file by running

.. code-block:: bash

    reno new your-short-descriptor-here

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.

See reno's `user guide <https://docs.openstack.org/reno/latest/user/usage.html>`_
for details.
