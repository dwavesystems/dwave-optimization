============
Contributing
============

Ocean's `contributing guide <https://docs.dwavequantum.com/en/latest/ocean/contribute.html>`_
has guidelines for contributing to Ocean packages. With the following changes

* ``dwave-optimization`` uses C++20.

pre-commit
==========

All changes must be formatted and checked by [ruff](https://docs.astral.sh/ruff/)
and [clang-format](https://clang.llvm.org/docs/ClangFormat.html).

To make this as easy as possible, you can setup [pre-commit](https://pre-commit.com/):

```bash
pip install --group pre-commit
pre-commit install
```

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
