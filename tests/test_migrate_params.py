from twitter import migrate_params


def test_v0_to_v1():
    assert migrate_params(
        {
            "querytype": 1,
            "query": "foo",
            "username": "",
            "listurl": "",
            "update": "",
            "accumulate": False,
        }
    ) == {
        "querytype": "search",
        "query": "foo",
        "username": "",
        "listurl": "",
        "update": "",
        "accumulate": False,
    }


def test_v1():
    expected = {
        "querytype": "lists_statuses",
        "query": "foo",
        "username": "",
        "listurl": "http://foo",
        "update": "",
        "accumulate": False,
    }
    assert migrate_params(expected) == expected
