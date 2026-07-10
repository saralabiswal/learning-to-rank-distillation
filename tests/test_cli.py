import pytest

from learning_to_rank_distillation.cli import main


def test_cli_train_teacher_exposes_all_dataset_choices(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["train-teacher", "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "{synthetic,esci,rectour,movielens}" in output


def test_cli_benchmark_exposes_dataset_switch(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["benchmark", "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--dataset" in output
    assert "--data-dir" in output


def test_cli_distillation_ablation_exposes_method_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["distillation-ablation", "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "response-KD" in output
    assert "--teacher-epochs" in output
