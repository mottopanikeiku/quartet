"""Report rendering, artifact IO, and experiment verification utilities.

The tuner algorithms remain in :mod:`quartet.bandit`, :mod:`quartet.retrieval`,
:mod:`quartet.replay`, :mod:`quartet.multisource_engine`, and
:mod:`quartet.v3_engine`. Import tooling from its individual modules; importing
this package does not load report renderers, verifiers, or subprocess workers.
"""
