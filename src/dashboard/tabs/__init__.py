"""Modular tab packages for the QC Monitor dashboard.

Each tab module exports ``layout(store)`` and ``register_callbacks(app, store, config)``.
The main app factory wires them up by importing from this package.
"""
