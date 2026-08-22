"""Experiment instruments: one Consumer per module.

A Consumer turns prepared data into a scalar loss.  The forecasting Consumer
has always lived inline in the batch runners (``bch._evaluate_assignment``);
Phase T's second Consumer lands here.  This package is deliberately outside
``methods/ttha``: a Consumer is the experiment's measuring instrument, not a
part of the Harness under study, and nothing here may be imported by the
Harness.
"""
