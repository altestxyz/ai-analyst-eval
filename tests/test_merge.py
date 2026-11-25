import json

from aae import merge, restore


def rec(rid, model="F1", ctx="L1", ts="2026-11-18T00:00:00", **extra):
    return {"run_id": rid, "question_id": rid.split("|")[0], "model": model, "context": ctx, "timestamp": ts, **extra}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_merge_counts_each_run_once_across_sessions(tmp_path):
    a = [rec("1|L1|F1|0"), rec("2|L1|F1|0")]
    b = [rec("2|L1|F1|0"), rec("3|L1|F1|0")]
    write(tmp_path / "run_a" / "runs.jsonl", a)
    write(tmp_path / "run_b.jsonl", b)
    write(tmp_path / "copy_of_b.jsonl", b)
    records, manifest = merge.merge(tmp_path)
    assert [r["run_id"] for r in records] == ["1|L1|F1|0", "2|L1|F1|0", "3|L1|F1|0"]
    assert manifest["unique_runs"] == 3 and manifest["conflicts"] == 0
    assert manifest["identical_duplicates"] == 3


def test_conflicting_copies_keep_the_first_run_by_default(tmp_path):
    write(tmp_path / "a.jsonl", [rec("1|L1|F1|0", ts="2026-11-18T01:00:00", correct=True)])
    write(tmp_path / "b.jsonl", [rec("1|L1|F1|0", ts="2026-11-19T01:00:00", correct=False)])
    records, manifest = merge.merge(tmp_path)
    assert manifest["conflicts"] == 1 and records[0]["correct"] is True
    latest, _ = merge.merge(tmp_path, prefer="latest")
    assert latest[0]["correct"] is False


def test_label_files_and_samples_are_not_treated_as_run_logs(tmp_path):
    write(tmp_path / "runs.jsonl", [rec("1|L1|F1|0")])
    write(tmp_path / "shape.jsonl", [{"run_id": "1|L1|F1|0", "shape": "empty"}])
    write(tmp_path / "sample.jsonl", [rec("9|L1|F1|0")])
    (tmp_path / "bad.jsonl").write_text("not json\n")
    records, _ = merge.merge(tmp_path)
    assert [r["run_id"] for r in records] == ["1|L1|F1|0"]


def test_half_written_lines_are_skipped(tmp_path):
    (tmp_path / "runs.jsonl").write_text(json.dumps(rec("1|L1|F1|0")) + "\n" + '{"run_id": "2|L1|F1|0", "mod')
    records, _ = merge.merge(tmp_path)
    assert len(records) == 1


def test_different_plans_are_reported(tmp_path):
    write(tmp_path / "s1" / "runs" / "runs.jsonl", [rec("1|L1|F1|0")])
    write(tmp_path / "s2" / "runs" / "runs.jsonl", [rec("2|L1|F1|0")])
    (tmp_path / "s1" / "artifacts").mkdir()
    (tmp_path / "s2" / "artifacts").mkdir()
    (tmp_path / "s1" / "artifacts" / "frozen.json").write_text(json.dumps({"parts": {"prompt": "a"}}))
    (tmp_path / "s2" / "artifacts" / "frozen.json").write_text(json.dumps({"parts": {"prompt": "b"}}))
    _, manifest = merge.merge(tmp_path)
    assert manifest["plan_consistent"] is False


def test_write_records_a_checksum(tmp_path):
    write(tmp_path / "in" / "a.jsonl", [rec("1|L1|F1|0")])
    records, manifest = merge.merge(tmp_path / "in")
    merge.write(records, manifest, tmp_path / "out" / "runs.jsonl", tmp_path / "out" / "manifest.json")
    saved = json.load(open(tmp_path / "out" / "manifest.json"))
    assert saved["output"]["lines"] == 1 and len(saved["output"]["sha256"]) == 64


def test_restore_merges_saved_sessions_and_copies_the_frozen_files(tmp_path):
    for name, rid in (("s1", "1|L1|F1|0"), ("s2", "2|L1|F1|0")):
        write(tmp_path / "in" / name / "runs" / "runs.jsonl", [rec(rid), rec("shared|L1|F1|0")])
        (tmp_path / "in" / name / "artifacts").mkdir(parents=True)
        (tmp_path / "in" / name / "artifacts" / "questions.jsonl").write_text("{}\n")
    summary = restore.restore([tmp_path / "in"], tmp_path / "dest")
    assert summary["merged"]["runs.jsonl"] == 3
    assert (tmp_path / "dest" / "artifacts" / "questions.jsonl").exists()
