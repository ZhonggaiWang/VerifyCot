"""Regression tests for the phase-three package boundaries."""

import importlib
import importlib.util
import json
import subprocess
import sys


def test_binary_mainline_does_not_eagerly_import_four_way():
    code = r'''
import json
import sys
import grounding_control
import grounding_control.core
import grounding_control.verifiers
import grounding_control.verifiers.qwen25_vl
import grounding_control.workers.dino_grounder
import grounding_control.workers.dino_verifier
import grounding_control.workers.qwen_grounder
import grounding_control.workers.qwen_verifier
import model.load_model
print(json.dumps(sorted(
    name for name in sys.modules
    if name.startswith("grounding_control.four_way")
)))
'''
    completed = subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout.strip()) == []


def test_non_mainline_namespaces_are_explicit_and_importable():
    four_way = importlib.import_module('grounding_control.four_way')
    legacy = importlib.import_module('grounding_control.legacy')

    assert four_way.FourWayPrecommitGroundingController.__module__ == (
        'grounding_control.four_way.controller'
    )
    assert four_way.RoutingPolicy.__module__ == (
        'grounding_control.four_way.routing_policy'
    )
    assert legacy.LegacyRepairController.__module__ == (
        'grounding_control.legacy.repair_controller'
    )
    assert legacy.VerificationResult.__module__ == (
        'grounding_control.legacy.verdicts'
    )


def test_removed_facades_and_duplicate_modules_are_absent():
    removed = (
        'verifier',
        'grounding_control.routing_controller',
        'grounding_control.routing_policy',
        'grounding_control.expert_router',
        'grounding_control.runtime',
        'grounding_control.verifier_backends',
        'grounding_control.prompts',
        'grounding_control.types',
        'grounding_control.stored_oracle',
        'grounding_control.single_candidate_oracle',
    )
    for module_name in removed:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_canonical_worker_modules_are_importable():
    dino_grounder = importlib.import_module(
        'grounding_control.workers.dino_grounder'
    )
    binary_dino = importlib.import_module(
        'grounding_control.workers.dino_verifier'
    )
    qwen_grounder = importlib.import_module(
        'grounding_control.workers.qwen_grounder'
    )
    binary_qwen = importlib.import_module(
        'grounding_control.workers.qwen_verifier'
    )
    four_way_dino = importlib.import_module(
        'grounding_control.four_way.workers.dino_geometry_verifier'
    )
    four_way_qwen = importlib.import_module(
        'grounding_control.four_way.workers.qwen_verifier_dino_grounder'
    )

    assert callable(dino_grounder.main)
    assert callable(binary_dino.main)
    assert callable(qwen_grounder.main)
    assert callable(binary_qwen.main)
    assert callable(four_way_dino.main)
    assert callable(four_way_qwen.main)
