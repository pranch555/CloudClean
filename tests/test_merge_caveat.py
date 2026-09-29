"""A merge flagged by its pre-merge check keeps that warning for every measurement made on it or its children."""
from cloudclean.register import merge_caveat


def test_no_caveat_for_a_clean_merge():
    assert merge_caveat({"scans": [{"index": 0}, {"index": 1, "fitness": 0.9}]}) is None
    clean = {"scans": [{"index": 0}, {"index": 1}],
             "assessment": {"scans": [{"index": 0}, {"index": 1, "ambiguous": False, "doubled_surface": False}]}}
    assert merge_caveat(clean) is None


def test_ambiguous_and_doubled_merge_is_flagged():
    report = {"scans": [{"index": 0}, {"index": 1}],
              "assessment": {"scans": [{"index": 0}, {"index": 1, "ambiguous": True, "alternative_angle": -30.4,
                                                      "doubled_surface": True, "layering_mm": 0.0763}]}}
    c = merge_caveat(report)
    assert c["ambiguous"] and c["doubled_surface"]
    assert c["alternative_angle"] == 30.4 and c["layering_mm"] == 0.0763
    assert "30°" in c["sentence"] and "0.08 mm" in c["sentence"]


def test_automatic_alignment_flag_alone_is_enough():
    c = merge_caveat({"scans": [{"index": 0}, {"index": 1, "ambiguous": True}]})
    assert c["ambiguous"] and not c["doubled_surface"] and c["alternative_angle"] is None
