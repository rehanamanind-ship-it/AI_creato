import argparse

from app import AutoLearn, build_parser, parse_equals_input


def test_parse_equals_input_extracts_roles():
    text = """
=SYSTEM= You are helpful
=USER= Hello there
=ASSISTANT= Hi!
===
=USER= Second prompt
=ASSISTANT= Second answer
"""
    parsed = parse_equals_input(text)

    assert len(parsed) == 2
    assert parsed[0]["system"] == "You are helpful"
    assert parsed[0]["user"] == "Hello there"
    assert parsed[0]["assistant"] == "Hi!"
    assert parsed[1]["user"] == "Second prompt"


def test_auto_learn_selects_llm_for_long_outputs():
    learner = AutoLearn()
    rows = [{"output": "word " * 20}, {"output": "word " * 30}]

    assert learner.choose_model(rows) == "LLM"


def test_build_parser_has_train_command():
    parser = build_parser()

    args = parser.parse_args(["train", "training.txt"])
    assert args.command == "train"
    assert args.input_file == "training.txt"


def test_build_parser_has_fine_tune_and_test_commands():
    parser = build_parser()

    fine_tune_args = parser.parse_args(["fine-tune", "examples.txt", "--base-model", "distilgpt2"])
    assert fine_tune_args.command == "fine-tune"
    assert fine_tune_args.training_file == "examples.txt"

    test_args = parser.parse_args(["test", "model_dir", "--prompt", "hello there"])
    assert test_args.command == "test"
    assert test_args.model_dir == "model_dir"
    assert test_args.prompt_flag == "hello there"


def test_build_parser_supports_routing_and_selection():
    parser = build_parser()

    args = parser.parse_args([
        "train",
        "training.txt",
        "--route",
        "sklearn",
        "--model",
        "logistic-regression",
    ])
    assert args.route == "sklearn"
    assert args.model == "logistic-regression"
