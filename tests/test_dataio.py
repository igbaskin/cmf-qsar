"""What the exported files have to carry."""


def test_predictions_csv_carries_the_applicability_domain(tmp_path):
    """An extrapolation has to stay marked after the predictions leave the program."""
    import csv
    from cmfqsar.dataio import write_predictions_csv

    path = tmp_path / "pred.csv"
    write_predictions_csv(str(path), ["a", "b", "c"], [1.0, 2.0, float("nan")],
                          [1.1, 2.2, 3.3], ["train", "test", "test"],
                          domain=[float("nan"), 0.95, 0.40], domain_threshold=0.8)
    rows = list(csv.DictReader(open(path)))
    assert rows[0]["ad_similarity"] == "" and rows[0]["in_domain"] == ""
    assert rows[1]["ad_similarity"] == "0.9500" and rows[1]["in_domain"] == "yes"
    assert rows[2]["in_domain"] == "no"


def test_predictions_csv_without_a_domain_keeps_its_old_columns(tmp_path):
    import csv
    from cmfqsar.dataio import write_predictions_csv

    path = tmp_path / "plain.csv"
    write_predictions_csv(str(path), ["a"], [1.0], [1.1], ["train"])
    header = next(csv.reader(open(path)))
    assert header == ["name", "subset", "experimental", "predicted"]
