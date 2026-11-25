import argparse

from aae import pipeline


def test_taxi_step_is_skipped_when_the_set_is_not_present(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline, "download", lambda *a: calls.append(a))
    monkeypatch.setattr(pipeline, "mod", lambda *a, **k: calls.append(a))
    pipeline.cmd_taxi(argparse.Namespace(months=[1, 2, 3]))
    assert calls == []
    assert "Skipping the taxi set" in capsys.readouterr().out


def test_taxi_step_copies_a_supplied_set_before_building(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "taxi" / "database_description").mkdir(parents=True)
    (tmp_path / "taxi" / "taxi_questions.jsonl").write_text("{}\n")
    (tmp_path / "taxi" / "database_description" / "trips.csv").write_text("original_column_name\n")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "questions.jsonl").write_text("")
    built = []
    monkeypatch.setattr(pipeline, "download", lambda *a: None)
    monkeypatch.setattr(pipeline, "mod", lambda name, *a, **k: built.append(name))
    pipeline.cmd_taxi(argparse.Namespace(months=[1]))
    assert built == ["build_taxi", "add_taxi"]
    assert (tmp_path / "artifacts" / "taxi_questions.jsonl").exists()
    assert (tmp_path / "data" / "taxi" / "database_description" / "trips.csv").exists()
