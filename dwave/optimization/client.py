#
# DO NOT INCLUDE THIS FILE IN THE FINAL DWAVE-OPTIMIZATION PACKAGE!
#

################################################################################
#
# dwave-cloud-client
#

"""
This file adds support for NL solver to `dwave-cloud-client` and `dwave-system`
Ocean packages, while we are in the "stealth mode" (as Ocean packages are OSS).
"""

import io
import sys
import concurrent.futures
from tempfile import SpooledTemporaryFile
from typing import Optional, Union

import dwave.cloud.computation
from dwave.cloud import logger
from dwave.cloud.solver import BaseUnstructuredSolver, available_solvers


class NLSolver(BaseUnstructuredSolver):
    """NL solver interface.

    This class provides an :term:`NL model` sampling method and encapsulates
    the solver description returned from the D-Wave cloud API.

    Args:
        client (:class:`~dwave.cloud.client.Client`):
            Client that manages access to this solver.

        data (`dict`):
            Data from the server describing this solver.

    Note:
        Events are not yet dispatched from unstructured solvers.
    """

    _handled_problem_types = {"nl"}
    _handled_encoding_formats = {"binary-ref"}

    def upload_problem(self, problem, **kwargs):
        # TODO: move this to `BaseUnstructuredSolver` once we go public
        data = self._encode_problem_for_upload(problem, **kwargs)
        return self.client.upload_problem_encoded(data)

    def _encode_problem_for_upload(self,
                                   model: Union['dwave.optimization.Model', io.IOBase],
                                   **kwargs
                                   ) -> io.IOBase:
        try:
            data = model.to_file(**kwargs)
        except Exception as e:
            logger.debug("NL model serialization failed with %r, "
                         "assuming data already encoded.", e)
            # assume `model` given as file, ready for upload
            data = model

        logger.debug("Problem (model) encoded for upload: %r", data)
        return data

    # TODO: add max_num_states before we go public -- we need to support
    # `upload_params` and `sample_params`
    def sample_nlm(self, model: Union['dwave.optimization.Model', io.IOBase, str],
                   label: Optional[str] = None, **params) -> dwave.cloud.computation.Future:
        """Sample from the specified :term:`NL model`.

        Args:
            model (:class:`~dwave.optimization.Model`/bytes/str):
                A nonlinear model, serialized model, or a reference to uploaded
                model (Problem ID returned by `.upload_nlm` method).

            label (str, optional):
                Problem label you can optionally tag submissions with for ease
                of identification.

            **params:
                Parameters for the sampling method, solver-specific.

        Returns:
            :class:`~dwave.cloud.computation.Future`

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.sample_problem(model, label=label, **params)

    def sample_problem(self, *args, **kwargs):
        sf = SpooledTemporaryFile(max_size=1e8, mode='w+b')
        # backport a fix for bpo-26175, https://github.com/python/cpython/pull/29560.
        # to make sure `zipfile` works with `SpooledTemporaryFile` in Python < 3.11
        if not hasattr(sf, 'seekable'):
            # of all the methods fixed in the above PR, only seekable is actually
            # ever used in `zipfile`.
            sf.seekable = lambda: sf._file.seekable()

        future = super().sample_problem(*args, **kwargs)
        future._answer_data = sf

        return future

    def upload_nlm(self,
                   model: Union['dwave.optimization.Model', io.IOBase, bytes],
                   **kwargs,
                   ) -> concurrent.futures.Future:
        """Upload the specified :term:`NL model` to SAPI, returning a Problem ID
        that can be used to submit the NL model to this solver (i.e. call the
        :meth:`.sample_nlm` method).

        Args:
            model (:class:`~dwave.optimization.Model`/bytes-like/file-like):
                A nonlinear model given either as an in-memory
                :class:`~dwave.optimization.Model` object, or as raw data
                (encoded serialized model) in either a file-like or a bytes-like
                object.
            max_num_states (int):
                Maximum number of states to upload along with the model.
                The number of states uploaded is ``min(len(model.states), max_num_states)``.

        Returns:
            :class:`concurrent.futures.Future`[str]:
                Problem ID in a Future. Problem ID can be used to submit
                problems by reference.

        Note:
            To use this method, dimod package has to be installed.
        """
        return self.upload_problem(model, **kwargs)

sys.modules['dwave.cloud.solver'].NLSolver = NLSolver

available_solvers.append(NLSolver)


################################################################################
#
# dwave-system
#

import concurrent.futures
import warnings
from collections import abc
from typing import Any, Dict, List, NamedTuple, Optional

import dwave.system.samplers
from dwave.cloud.client import Client
from dwave.system.utilities import classproperty, FeatureFlags


class LeapHybridNLSampler:
    """A class for using Leap's cloud-based hybrid nonlinear-model solvers.

    Leap's quantum-classical hybrid nonlinear-model solvers are intended to 
    solve application problems formulated as
    :ref:`nonlinear models <nl_model_sdk>`.

    You can configure your :term:`solver` selection and usage by setting 
    parameters, hierarchically, in a configuration file, as environment 
    variables, or explicitly as input arguments, as described in
    `D-Wave Cloud Client <https://docs.ocean.dwavesys.com/en/stable/docs_cloud/sdk_index.html>`_.

    :ref:`dwave-cloud-client <sdk_index_cloud>`'s
    :meth:`~dwave.cloud.client.Client.get_solvers` method filters solvers you 
    have access to by 
    `solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
    ``category=hybrid`` and ``supported_problem_type=nl``. By default, online
    hybrid nonlinear-model solvers are returned ordered by latest ``version``.

    Args:
        **config:
            Keyword arguments passed to :meth:`dwave.cloud.client.Client.from_config`.

    Examples:
        This example submits a model for a 
        :class:`flow-shop-scheduling <dwave.optimization.generators.flow_shop_scheduling>`
        problem. 
    
        >>> from dwave.optimization.generators import flow_shop_scheduling
        >>> from dwave.system import LeapHybridNLSampler 
        ...
        >>> sampler = LeapHybridNLSampler()     # doctest: +SKIP
        ...
        >>> processing_times = [[10, 5, 7], [20, 10, 15]]
        >>> model = flow_shop_scheduling(processing_times=processing_times) 
        >>> results = sampler.sample(model, label="Small FSS problem")    # doctest: +SKIP
        >>> job_order = next(model.iter_decisions())  # doctest: +SKIP
        >>> print(f"State 0 of {model.objective.state_size()} has an "\   # doctest: +SKIP
        ... f"objective value {model.objective.state(0)} for order " \    # doctest: +SKIP
        ... f"{job_order.state(0)}.")     # doctest: +SKIP
        State 0 of 8 has an objective value 50.0 for order [1. 2. 0.]. 
    """

    def __init__(self, **config):
        # strongly prefer hybrid solvers; requires kwarg-level override
        config.setdefault('client', 'hybrid')

        # default to short-lived session to prevent resets on slow uploads
        config.setdefault('connection_close', True)

        if FeatureFlags.hss_solver_config_override:
            # use legacy behavior (override solver config from env/file)
            solver = config.setdefault('solver', {})
            if isinstance(solver, abc.Mapping):
                solver.update(self.default_solver)

        # prefer the latest hybrid NL solver available, but allow for an easy
        # override on any config level above the defaults (file/env/kwarg)
        defaults = config.setdefault('defaults', {})
        if not isinstance(defaults, abc.Mapping):
            raise TypeError("mapping expected for 'defaults'")
        defaults.update(solver=self.default_solver)

        self.client = Client.from_config(**config)
        self.solver = self.client.get_solver()

        # For explicitly named solvers:
        if self.properties.get('category') != 'hybrid':
            raise ValueError("selected solver is not a hybrid solver.")
        if 'nl' not in self.solver.supported_problem_types:
            raise ValueError("selected solver does not support the 'nl' problem type.")

        self._executor = concurrent.futures.ThreadPoolExecutor()

    @classproperty
    def default_solver(cls) -> Dict[str, str]:
        """Features used to select the latest accessible hybrid CQM solver."""
        return dict(supported_problem_types__contains='nl',
                    order_by='-properties.version')

    @property
    def properties(self) -> Dict[str, Any]:
        """Solver properties as returned by a SAPI query.

        `Solver properties <https://docs.dwavesys.com/docs/latest/c_solver_properties.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._properties
        except AttributeError:
            self._properties = properties = self.solver.properties.copy()
            return properties

    @property
    def parameters(self) -> Dict[str, List[str]]:
        """Solver parameters in the form of a dict, where keys
        are keyword parameters accepted by a SAPI query and values are lists of
        properties in
        :attr:`~dwave.system.samplers.LeapHybridNLSampler.properties` for each
        key.

        `Solver parameters <https://docs.dwavesys.com/docs/latest/c_solver_parameters.html>`_
        are dependent on the selected solver and subject to change.
        """
        try:
            return self._parameters
        except AttributeError:
            parameters = {param: ['parameters']
                          for param in self.properties['parameters']}
            parameters.update(label=[])
            self._parameters = parameters
            return parameters

    class SampleResult(NamedTuple):
        model: 'dwave.optimization.Model'
        timing: dict

    def sample(self, model: 'dwave.optimization.Model',
               time_limit: Optional[float] = None, **kwargs
               ) -> 'concurrent.futures.Future[SampleResult]':
        """Sample from the specified nonlinear model.

        Args:
            model (:class:`~dwave.optimization.Model`):
                Nonlinear model.

            time_limit (float, optional):
                Maximum run time, in seconds, to allow the solver to work on the
                problem. Must be at least the minimum required for the problem,
                which is calculated and set by default.

                :meth:`~dwave.system.samplers.LeapHybridNLMSampler.estimated_min_time_limit`
                estimates (and describes) the minimum time for your problem.

            **kwargs:
                Optional keyword arguments for the solver, specified in
                :attr:`~dwave.system.samplers.LeapHybridNLMSampler.parameters`.

        Returns:
            :class:`concurrent.futures.Future`[SampleResult]:
                Named tuple containing nonlinear model and timing info, in a Future.
        """

        # TODO: move to global context in dwave-system
        from dwave.optimization import Model

        if not isinstance(model, Model):
            raise TypeError("first argument 'model' must be a dwave.optimization.Model, "
                            f"received {type(model).__name__}")

        if time_limit is None:
            time_limit = self.estimated_min_time_limit(model)

        num_states = len(model.states)
        max_num_states = min(
            self.solver.properties.get("maximum_number_of_states", num_states),
            num_states)
        problem_data_id = self.solver.upload_nlm(model, max_num_states=max_num_states).result()

        future = self.solver.sample_nlm(problem_data_id, time_limit=time_limit, **kwargs)

        def hook(model, future):
            # TODO: known bug, don't check header for now
            model.states.from_file(future.answer_data, check_header=False)

        model.states.from_future(future, hook)

        def collect():
            timing = future.timing
            for msg in timing.get('warnings', []):
                # note: no point using stacklevel, as this is a different thread
                warnings.warn(msg, category=UserWarning)

            return LeapHybridNLSampler.SampleResult(model, timing)

        result = self._executor.submit(collect)

        return result

    def estimated_min_time_limit(self, nlm: 'dwave.optimization.Model') -> float:
        """Return the minimum `time_limit`, in seconds, accepted for the given problem."""

        num_nodes_multiplier = self.properties.get('num_nodes_multiplier', 8.306792043756981e-05)
        state_size_multiplier = self.properties.get('state_size_multiplier', 2.8379674360396316e-10)
        num_nodes_state_size_multiplier = self.properties.get('num_nodes_state_size_multiplier', 2.1097317822863966e-12)
        offset = self.properties.get('offset', 0.012671678446550175)
        min_time_limit = self.properties.get('min_time_limit', 5)

        nn = nlm.num_nodes()
        ss = nlm.state_size()

        return max(
            num_nodes_multiplier * nn
            + state_size_multiplier * ss
            + num_nodes_state_size_multiplier * nn * ss
            + offset,
            min_time_limit
        )

sys.modules['dwave.system'].LeapHybridNLSampler = LeapHybridNLSampler
sys.modules['dwave.system.samplers'].LeapHybridNLSampler = LeapHybridNLSampler
sys.modules['dwave.system.samplers.leap_hybrid_sampler'].LeapHybridNLSampler = LeapHybridNLSampler
