import pytest

from kvbench.benchmark.blocktrace import summarize_issues, validate_trace_log


def test_issue_order_sequential_runs_and_interleaving(tmp_path):
    path = tmp_path / "issues.txt"
    path.write_text("""1.0 D WS 100 8
1.1 D WS 108 8
1.2 C WS 100 8
1.3 D R 5 8
1.4 D WS 10000 16
1.5 D W 116 8
1.6 D WS 124 8
1.7 D WF 0 0
""")
    result = summarize_issues(path)
    assert result["write_ops"] == 5
    assert result["write_bytes"] == 48 * 512
    assert result["adjacent_write_transitions"] == 2
    assert result["sequential_transition_fraction"] == 0.5
    assert result["backward_write_transitions"] == 1
    assert result["max_contiguous_run_bytes"] == 8192
    assert result["unique_written_lba_bytes"] == 48 * 512
    assert result["revisited_write_bytes"] == 0


def test_partial_overlaps_duplicates_and_out_of_order_addresses(tmp_path):
    path = tmp_path / "issues.txt"
    path.write_text("1.0 D W 100 8\n1.1 D W 104 8\n1.2 D W 100 8\n1.3 D W 0 8\n")
    result = summarize_issues(path)
    assert result["write_bytes"] == 32 * 512
    assert result["unique_written_lba_bytes"] == 20 * 512
    assert result["revisited_write_bytes"] == 12 * 512
    assert result["revisited_write_fraction"] == 0.375
    assert not list(tmp_path.glob("kvbench-lba-*"))


def test_no_writes_has_no_revisit_fraction(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("1.0 D R 100 8\n")
    result = summarize_issues(path)
    assert result["unique_written_lba_bytes"] == result["revisited_write_bytes"] == 0
    assert result["revisited_write_fraction"] is None


def test_lba_coverage_across_database_batches_matches_sector_oracle(tmp_path):
    import random

    rng = random.Random(24301)
    spans = [(rng.randrange(5000), rng.randrange(1, 100)) for _ in range(5000)]
    path = tmp_path / "issues.txt"
    path.write_text("".join(f"{i}.0 D W {start} {length}\n" for i, (start, length) in enumerate(spans)))
    sectors = {sector for start, length in spans for sector in range(start, start + length)}
    total = sum(length for _, length in spans)
    result = summarize_issues(path)
    assert result["unique_written_lba_bytes"] == len(sectors) * 512
    assert result["revisited_write_bytes"] == (total - len(sectors)) * 512


def test_dropped_or_failed_trace_is_not_a_valid_lba_sample():
    validate_trace_log("Total: 123 events (dropped 0)")
    for text in ("Total: 123 events (dropped 2)", "FAILED to start thread", "no drop accounting"):
        with pytest.raises(RuntimeError):
            validate_trace_log(text)
