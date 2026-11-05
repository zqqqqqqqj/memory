"""Local path handling and the retained ablation definitions."""
import copy
import importlib
import json
import pkgutil
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from qrm.config import ROOT, load_config
from qrm.paths import apply_server_paths


CONFIGS = sorted(ROOT.glob('ablation_*/configs/*.yaml'))


@pytest.mark.parametrize('path', CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_all_ten_ablation_rows_preserve_the_protocol(path):
    cfg = load_config(path)
    assert cfg['stages']['memory']['steps'] == cfg['stages']['memory']['action_checkpoint_step'] == 60000
    assert cfg['stages']['memory']['min_prefix_decisions'] == 17
    assert cfg['stages']['action']['steps'] == 30000
    assert cfg['sft']['global_batch_size'] == 64
    assert cfg['ablation']['parameter_initialization'] == 'named_linear_v1'
    assert cfg['history']['objective'] == 'residual'
    assert cfg['memory']['merge_rule'] == 'ward'
    assert cfg['plan'] == 'README.md' and (ROOT/cfg['plan']).is_file()


def test_public_reference_is_shared_between_all_groups():
    first = load_config(ROOT/'ablation_memory_capacity/configs/k8.yaml')
    assert first == load_config(ROOT/'ablation_residual_memory/configs/residual128.yaml')
    assert first == load_config(ROOT/'ablation_optimization/configs/reference.yaml')


def test_local_paths_change_no_training_parameters(tmp_path, monkeypatch):
    cfg = yaml.safe_load((ROOT/'ablation_memory_capacity/configs/k8.yaml').read_text())
    before = copy.deepcopy(cfg)
    path = tmp_path/'paths.yaml'
    path.write_text(yaml.safe_dump({'dataset': {'root': 'dataset'}, 'benchmark': {'root': 'benchmark'},
                                   'model': {'openpi_root': 'openpi'}}))
    monkeypatch.setenv('QRM_SERVER_CONFIG', str(path))
    result = apply_server_paths(cfg, ROOT)
    assert result['dataset']['root'] == str(tmp_path/'dataset')
    assert result['benchmark']['root'] == str(tmp_path/'benchmark')
    assert result['model']['openpi_root'] == str(tmp_path/'openpi')
    for section in ('memory', 'history', 'sft', 'stages', 'training', 'validation', 'ablation'):
        assert result[section] == before[section]


@pytest.mark.parametrize('content', [{'sft': {'steps': 1}}, {'model': {'memory_tokens': 0}},
                                   {'dataset': {'revision': 'wrong'}}, {'benchmark': {'replan_steps': 1}}])
def test_server_local_file_cannot_override_experiment_definitions(tmp_path, monkeypatch, content):
    path = tmp_path/'invalid.yaml'
    path.write_text(yaml.safe_dump(content))
    monkeypatch.setenv('QRM_SERVER_CONFIG', str(path))
    with pytest.raises(ValueError):
        apply_server_paths({}, ROOT)


def test_explicit_missing_server_configuration_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv('QRM_SERVER_CONFIG', str(tmp_path/'missing.yaml'))
    with pytest.raises(FileNotFoundError):
        apply_server_paths({}, ROOT)


def test_retained_package_has_no_missing_internal_imports():
    import qrm
    for item in pkgutil.walk_packages(qrm.__path__, qrm.__name__+'.'):
        importlib.import_module(item.name)


def test_unrelated_workflows_and_artifacts_are_not_packaged():
    for path in ('src/qrm/rl', 'src/qrm/diagnostics'):
        assert not (ROOT/path).exists()
    probes = ['data/index.json', 'runs/output.json', 'checkpoints/model.pt', 'reports/result.json',
              'server.local.yaml', 'src/qrm/__pycache__/core.pyc']
    if (ROOT/'.git').is_dir():
        result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], input='\n'.join(probes)+'\n',
                                cwd=ROOT, check=True, capture_output=True, text=True)
        assert result.stdout.splitlines() == probes
    assert len(CONFIGS) == 10


@pytest.mark.parametrize('suite', ['ablation_memory_capacity', 'ablation_residual_memory', 'ablation_optimization'])
def test_variant_listing_needs_no_server_assets(suite):
    result = subprocess.run([sys.executable, str(ROOT/suite/'run.py'), '--list'],
                            check=True, capture_output=True, text=True)
    expected = len(json.loads((ROOT/suite/'experiments.json').read_text())['variants'])
    assert len(result.stdout.splitlines()) == expected
