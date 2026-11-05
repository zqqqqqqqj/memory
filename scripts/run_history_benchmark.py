"""Run the unchanged benchmark in the lightweight simulator environment.

The history package initializer imports torch, although its config validator
does not need it. Load that exact validator module without executing the
training package initializer. All config checks and benchmark logic remain
the existing implementations; no model package is installed in the simulator.
"""
import importlib.util
import sys

from qrm.config import ROOT


def prepare_history_validation():
    name = 'qrm.history.config'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/'src/qrm/history/config.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module


if __name__ == '__main__':
    prepare_history_validation()
    from run_benchmark import main
    main()
