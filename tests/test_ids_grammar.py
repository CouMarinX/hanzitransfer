from hanzitransfer.fusion.ids_grammar import (
    IDSTree,
    contains_base,
    is_valid_ids,
    layout_hint,
    parse_ids,
    score_ids_legality,
)


def test_ids_parsing_and_validation():
    tree = parse_ids("⿰木木")
    assert layout_hint(tree) == "⿰"
    assert is_valid_ids(tree)
    assert contains_base(tree, "木")

    bad = IDSTree("⿰", [IDSTree("木", []), IDSTree("⿱", [IDSTree("木", [])])])
    assert not is_valid_ids(bad)
    assert score_ids_legality(bad) < score_ids_legality(tree)

