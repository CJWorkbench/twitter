import contextlib
import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Union

import dateutil
import lz4.frame
import pyarrow as pa
import pyarrow.parquet
from cjwmodule.i18n import I18nMessage
from cjwmodule.testing.i18n import cjwmodule_i18n_message, i18n_message

import twitter


def dt(s):
    return dateutil.parser.parse(s, ignoretz=True)


@contextlib.contextmanager
def override_settings(**kwargs):
    old_args = {k: getattr(twitter, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(twitter, k, v)
        yield
    finally:
        for k, v in old_args.items():
            setattr(twitter, k, v)


@contextlib.contextmanager
def _temp_parquet_file(table: pa.Table) -> ContextManager[Path]:
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        pa.parquet.write_table(table, path, version="2.0", compression="SNAPPY")
        yield path


@contextlib.contextmanager
def _temp_json_path_lz4(
    name: str, path: Path, pax_headers: Dict[str, str]
) -> ContextManager[Tuple[str, Dict[str, str], bytes]]:
    yield name, lz4.frame.compress(path.read_bytes()), pax_headers


@contextlib.contextmanager
def _temp_json_lz4(
    name: str, data: Dict[str, Any], pax_headers: Dict[str, str]
) -> ContextManager[Tuple[str, Dict[str, str], bytes]]:
    yield name, lz4.frame.compress(twitter._utf8_json_encode(data)), pax_headers


@contextlib.contextmanager
def _temp_tarfile(
    monads: List[Callable[[], ContextManager[Union[Path, Tuple[str, bytes]]]]]
) -> ContextManager[Path]:
    with tempfile.NamedTemporaryFile() as tf:
        with tarfile.open(tf.name, "w") as tar:
            for monad in monads:
                with monad() as part:
                    if isinstance(part, Path):
                        tar.add(part, arcname="LEGACY.parquet")
                    else:
                        name, data, pax_headers = part
                        ti = tarfile.TarInfo(name)
                        ti.size = len(data)
                        ti.pax_headers.update(pax_headers.items())
                        tar.addfile(ti, io.BytesIO(data))
        yield Path(tf.name)


def _assert_render(
    fetch_result: Optional[twitter.FetchResult],
    expected_table: Optional[pa.Table] = None,
    expected_errors: List[I18nMessage] = [],
):
    with tempfile.NamedTemporaryFile() as tf:
        output_path = Path(tf.name)
        actual_errors = twitter.render(
            pa.table({}), {}, output_path, fetch_result=fetch_result
        )
        assert actual_errors == expected_errors
        if expected_table is None:
            assert output_path.read_bytes() == b""
        else:
            with pa.ipc.open_file(tf.name) as f:
                actual_table = f.read_all()
            assert actual_table.column_names == expected_table.column_names
            for output_column, expected_column in zip(
                actual_table.itercolumns(), expected_table.itercolumns()
            ):
                assert output_column.type == expected_column.type
                assert output_column.to_pylist() == expected_column.to_pylist()
                if pa.types.is_dictionary(output_column.type):
                    for output_chunk, expected_chunk in zip(
                        output_column.iterchunks(), expected_column.iterchunks()
                    ):
                        assert (
                            output_chunk.dictionary.to_pylist()
                            == expected_chunk.dictionary.to_pylist()
                        )


def test_render_empty_no_query():
    # When we haven't fetched, we shouldn't show any columns (for
    # consistency with other modules)
    _assert_render(None, None, [])


def test_render_fetch_generated_error():
    # fetch() generates errors when params are invalid
    _assert_render(
        twitter.FetchResult(Path("unused"), [i18n_message("error.invalidUsername")]),
        None,
        [i18n_message("error.invalidUsername")],
    )


def test_render_empty_tarfile():
    # When we haven't fetched, we shouldn't show any columns (for
    # consistency with other modules)
    with _temp_tarfile([]) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            twitter.ARROW_SCHEMA.empty_table(),
            [],
        )


def test_render_v0_zero_column_search_result():
    # An empty table might be stored as zero-column. This is a bug, but we
    # must handle it because we have actual data like this. We want to
    # output all the same columns as a tweet table.
    with _temp_parquet_file(pa.table({})) as parquet_path:
        _assert_render(
            twitter.FetchResult(parquet_path, []),
            twitter.ARROW_SCHEMA.empty_table(),
            [],
        )


def test_render_v0_empty_table():
    with _temp_parquet_file(twitter.ARROW_SCHEMA.empty_table()) as parquet_path:
        _assert_render(
            twitter.FetchResult(parquet_path, []),
            twitter.ARROW_SCHEMA.empty_table(),
            [],
        )


@override_settings(TWITTER_MAX_ROWS_PER_TABLE=1)
def test_render_v0_truncate_fetch_results():
    all_rows = pa.table(
        {
            "screen_name": ["TheTweepyTester", "TheTweepyTester"],
            "created_at": pa.array(
                [dt("2016-11-05T21:38:46Z"), dt("2016-11-05T21:37:13Z")],
                pa.timestamp("ns"),
            ),
            "text": ["Hello", "testing 1000 https://t.co/3vt8ITRQ3w"],
            "retweet_count": [0, 0],
            "favorite_count": [0, 0],
            "in_reply_to_screen_name": pa.array([None, None], pa.utf8()),
            "retweeted_status_screen_name": pa.array([None, None], pa.utf8()),
            "user_description": ["", ""],
            "source": ["Twitter Web Client", "Tweepy dev"],
            "lang": ["en", "en"],
            "id": [795017539831103489, 795017147651162112],
        }
    )
    with _temp_parquet_file(all_rows) as parquet_path:
        _assert_render(twitter.FetchResult(parquet_path, []), all_rows.slice(0, 1), [])


def test_render_v0_recover_after_bug_160258591():
    # https://www.pivotaltracker.com/story/show/160258591
    # 'id', 'retweet_count' and 'favorite_count' had wrong type after
    # accumulating an empty table. Now the bad data is in our database;
    # let's convert back to the type we want.
    input_table = pa.table(
        {
            "screen_name": ["TheTweepyTester", "TheTweepyTester"],
            "created_at": pa.array(
                [dt("2016-11-05T21:38:46Z"), dt("2016-11-05T21:37:13Z")],
                pa.timestamp("ns"),
            ),
            "text": ["Hello", "testing 1000 https://t.co/3vt8ITRQ3w"],
            "retweet_count": ["0", "0"],
            "favorite_count": ["0", "0"],
            "in_reply_to_screen_name": pa.array([None, None], pa.utf8()),
            "retweeted_status_screen_name": pa.array([None, None], pa.utf8()),
            "user_description": ["", ""],
            "source": ["Twitter Web Client", "Tweepy dev"],
            "lang": ["en", "en"],
            "id": ["795017539831103489", "795017147651162112"],
        }
    )
    with _temp_parquet_file(input_table) as parquet_path:
        _assert_render(
            twitter.FetchResult(parquet_path, []),
            (
                input_table.set_column(3, "retweet_count", pa.array([0, 0]))
                .set_column(4, "favorite_count", pa.array([0, 0]))
                .set_column(
                    10, "id", pa.array([795017539831103489, 795017147651162112])
                )
            ),
            [],
        )


def test_render_v0_add_retweet_status_screen_name():
    # Migration: what happens when we accumulate tweets
    # where the old stored table does not have retweet_status_screen_name?
    # We should consider those to have just None in that column
    input_table = pa.table(
        {
            "screen_name": ["TheTweepyTester", "TheTweepyTester"],
            "created_at": pa.array(
                [dt("2016-11-05T21:38:46Z"), dt("2016-11-05T21:37:13Z")],
                pa.timestamp("ns"),
            ),
            "text": ["Hello", "testing 1000 https://t.co/3vt8ITRQ3w"],
            "retweet_count": [0, 0],
            "favorite_count": [0, 0],
            "in_reply_to_screen_name": pa.array([None, None], pa.utf8()),
            "user_description": ["", ""],
            "source": ["Twitter Web Client", "Tweepy dev"],
            "lang": ["en", "en"],
            "id": [795017539831103489, 795017147651162112],
        }
    )
    with _temp_parquet_file(input_table) as parquet_path:
        _assert_render(
            twitter.FetchResult(parquet_path, []),
            input_table.add_column(
                6, "retweeted_status_screen_name", pa.array([None, None], pa.utf8())
            ),
            [],
        )


def test_render_retweeted_status_full_text_twitter_api_v1():
    with _temp_tarfile(
        [
            lambda: _temp_json_path_lz4(
                "1105492514289512400.json.lz4",
                Path("tests/files/1_1_one_extended_retweet.json"),
                {"cjw:apiEndpoint": "1.1/statuses/user_timeline.json"},
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            pa.table(
                {
                    "screen_name": ["workbenchdata"],
                    "created_at": pa.array(
                        [dt("Tue Mar 12 15:35:29 +0000 2019")], pa.timestamp("ns")
                    ),
                    "text": [
                        # "text" is the key data point we're testing
                        "RT @JacopoOttaviani: ⚡️ I'm playing with @workbenchdata: absolutely mindblowing. It's like a fusion between ScraperWiki, OpenRefine and Datawrapper. All of it online in the cloud and for free 👉🏽 https://t.co/fleqjI1qCI https://t.co/mmWHJLDjT2 #ddj #dataviz"
                    ],
                    "retweet_count": [7],
                    "favorite_count": [0],
                    "in_reply_to_screen_name": pa.nulls(1, pa.utf8()),
                    "retweeted_status_screen_name": ["JacopoOttaviani"],
                    "user_description": [
                        "Scrape, clean and analyze data without code. Create reproducible data workflows that can be shared with others"
                    ],
                    "source": ["Twitter for iPhone"],
                    "lang": ["en"],
                    "id": [1105492514289512400],
                }
            ),
            [],
        )


def test_render_undefined_language_is_null():
    # https://blog.twitter.com/developer/en_us/a/2013/introducing-new-metadata-for-tweets.html
    with _temp_tarfile(
        [
            lambda: _temp_json_path_lz4(
                "1088215462867959800.json.lz4",
                Path("tests/files/1_1_one_undefined_lang.json"),
                {"cjw:apiEndpoint": "1.1/statuses/user_timeline.json"},
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            pa.table(
                {
                    "screen_name": ["workbenchdata"],
                    "created_at": pa.array(
                        [dt("Wed Jan 23 23:22:39 +0000 2019")], pa.timestamp("ns")
                    ),
                    "text": ["🤖 https://t.co/FOhOfZT9MZ"],
                    "retweet_count": [0],
                    "favorite_count": [1],
                    "in_reply_to_screen_name": pa.nulls(1, pa.utf8()),
                    "retweeted_status_screen_name": pa.nulls(1, pa.utf8()),
                    "user_description": [
                        "Scrape, clean and analyze data without code. Create reproducible data workflows that can be shared with others"
                    ],
                    "source": ["Twitter for iPhone"],
                    # "lang" is the key data point we're testing
                    "lang": pa.nulls(1, pa.utf8()),
                    "id": [1088215462867959800],
                }
            ),
            [],
        )


def test_v2_one_sample_search_api_response():
    with _temp_tarfile(
        [
            lambda: _temp_json_path_lz4(
                "1332344846833639425.json.lz4",
                Path("tests/files/2_tweets_search_recent_page_1.json"),
                {"cjw:apiEndpoint": "2/tweets/search/recent"},
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            pa.table(
                {
                    "screen_name": [
                        "stmanfr",
                        "dutchscientist",
                        "Golden_Seagul",
                        "brian_thiede",
                        "LyingTruth2020",
                        "KalElSkywalker",
                        "FAX_online",
                        "K12Prospects",
                        "whitmer_joshua",
                        "AugustEichel",
                    ],
                    "created_at": pa.array(
                        [
                            dt("2020-11-27T15:25:39.000Z"),
                            dt("2020-11-27T15:25:39.000Z"),
                            dt("2020-11-27T15:25:38.000Z"),
                            dt("2020-11-27T15:25:37.000Z"),
                            dt("2020-11-27T15:25:37.000Z"),
                            dt("2020-11-27T15:25:37.000Z"),
                            dt("2020-11-27T15:25:36.000Z"),
                            dt("2020-11-27T15:25:36.000Z"),
                            dt("2020-11-27T15:25:34.000Z"),
                            dt("2020-11-27T15:25:34.000Z"),
                        ],
                        pa.timestamp("ns"),
                    ),
                    "text": [
                        "RT @ZinaAntoaneta: @NilsMelzer I'm on awe of your awareness of the #art &amp; #science of #propaganda. \nFor this is the battlefield of today's wars. \nWith the bloodiest consequences for so many #TargetedIndividuals.\nWe desperately need people of courage &amp; integrity. \nGod bless you, \n@NilsMelzer",
                        "@emmakennytv @MikeStuchbery_ And yes, I am a scientist working on infectious diseases. You know, the people who do the SCIENCE. Or science, as we call it as we are not requiring all-caps like drooling morons.\n\nYou are wrong and dangerously deluded.",
                        "RT @svaradarajan: Why AstraZeneca Is Facing Tricky Questions About Its COVID-19 Vaccine https://t.co/FaGFKBR9RX via @TheWireScience",
                        'New paper, "Climate variability and child nutrition: Findings from sub-Saharan Africa", with @JohannStrube, now online at GEC: https://t.co/OYPNGBWS2N.',
                        "@Backswimmer @BaronBeeGangsta @EricRWeinstein Inside the mind of a liberal most words have whatever meaning they want at that moment.\n\nDefund the police doesn’t mean defund the police, male doesn’t mean male, science doesn’t mean science, choice doesn’t mean choice.",
                        'RT @icymi_r: ✍️🧹 "Detect Relationships With Linear Regression (10 Must-Know Tidyverse Functions #4)"\n\n👤 Business Science @bizscienc; Matt Dancho @mdancho84 \n\nhttps://t.co/SkowBfjqLw\n#rstats https://t.co/YJ8vZ9FwUl',
                        'Brethren ...\n\nLet us pause for a moment and use our God-given Common Sense (and Charles Babbage-given Computer Science Knowledge) ...\n\nAnd inquire ...\n\nWhy would an "algorithm" be necessary for a computer program used for a simple tabulation?',
                        "RT @K12Prospects: Download A/B Test Planner https://t.co/Bkk2cEwTbJ\n#science #scienceteacher #Scratch3 #secondary #security https://t.co/qjFk3bwRwb",
                        "@Naname1961 @realDonaldTrump Math is difficult for people who don’t believe in science!",
                        "Science is fake, this is what I believe https://t.co/5SsyOQTa9A https://t.co/q4ZTq73pxh",
                    ],
                    "retweet_count": [1, 0, 2, 0, 0, 7, 0, 1, 0, 0],
                    "favorite_count": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "in_reply_to_screen_name": [
                        None,
                        "dutchscientist",
                        None,
                        None,
                        "Backswimmer",
                        None,
                        None,
                        None,
                        "Naname1961",
                        None,
                    ],
                    "retweeted_status_screen_name": [
                        "ZinaAntoaneta",
                        None,
                        "svaradarajan",
                        None,
                        None,
                        "icymi_r",
                        None,
                        "K12Prospects",
                        None,
                        None,
                    ],
                    # whoops, forgot to add "user.fields=description" to our stored API results.
                    # [adamhooper, 2020-11-27] you'll forgive me for not writing the whole test
                    # over again....
                    "user_description": pa.nulls(10, pa.utf8()),
                    "source": [
                        "Twitter for iPhone",
                        "Twitter Web App",
                        "Twitter for Android",
                        "Twitter Web App",
                        "Twitter for iPhone",
                        "Twitter for Android",
                        "Twitter Web App",
                        "Twitter Web App",
                        "Twitter Web App",
                        "Twitter for Android",
                    ],
                    "lang": [
                        "en",
                        "en",
                        "en",
                        "en",
                        "en",
                        "en",
                        "en",
                        "en",
                        "en",
                        "en",
                    ],
                    "id": [
                        1332344846833639425,
                        1332344843360677888,
                        1332344839682158592,
                        1332344838277230592,
                        1332344837794828289,
                        1332344837266432001,
                        1332344831478214662,
                        1332344831100772352,
                        1332344826206023681,
                        1332344825409048578,
                    ],
                }
            ),
            [],
        )


def test_render_network_error():
    with _temp_tarfile(
        [
            lambda: _temp_json_lz4(
                "NETWORK-ERROR.json.lz4",
                {
                    "id": "http.errors.HttpErrorGeneric",
                    "arguments": {"type": "NotImplemented"},
                    "source": "cjwmodule",
                },
                {"cjw:apiEndpoint": "2/tweets/search/recent"},
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            None,
            [
                cjwmodule_i18n_message(
                    "http.errors.HttpErrorGeneric", {"type": "NotImplemented"}
                )
            ],
        )


def test_render_http_429():
    with _temp_tarfile(
        [
            lambda: _temp_json_lz4(
                "API-ERROR.json.lz4",
                {"error": "doesn't really matter"},
                {"cjw:apiEndpoint": "2/tweets/search/recent", "cjw:httpStatus": "429"},
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            None,
            [i18n_message("error.tooManyRequests")],
        )


def test_render_http_401_user_tweets_are_private():
    with _temp_tarfile(
        [
            lambda: _temp_json_lz4(
                "API-ERROR.json.lz4",
                {"error": "doesn't really matter"},
                {
                    "cjw:apiEndpoint": "1.1/statuses/user_timeline",
                    "cjw:apiParams": "count=200&screen_name=elizabeth1",
                    "cjw:httpStatus": "401",
                },
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            None,
            [i18n_message("error.userTweetsArePrivate", {"username": "elizabeth1"})],
        )


def test_render_http_404_username_not_found():
    with _temp_tarfile(
        [
            lambda: _temp_json_lz4(
                "API-ERROR.json.lz4",
                {"error": "doesn't really matter"},
                {
                    "cjw:apiEndpoint": "1.1/statuses/user_timeline",
                    "cjw:apiParams": "count=200&screen_name=doesnotexistnoreally",
                    "cjw:httpStatus": "404",
                },
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            None,
            [
                i18n_message(
                    "error.userDoesNotExist", {"username": "doesnotexistnoreally"}
                )
            ],
        )


def test_render_v1_1_generic_api_error():
    with _temp_tarfile(
        [
            lambda: _temp_json_lz4(
                "API-ERROR.json.lz4",
                {"error": "a message from Twitter"},
                {
                    "cjw:apiEndpoint": "1.1/statuses/user_timeline",
                    "cjw:apiParams": "count=200&screen_name=adamhooper",
                    "cjw:httpStatus": "500",
                },
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            None,
            [
                i18n_message(
                    "error.genericApiErrorV1_1",
                    {"httpStatus": "500", "error": "a message from Twitter"},
                )
            ],
        )


def test_render_v2_generic_api_error():
    with _temp_tarfile(
        [
            lambda: _temp_json_lz4(
                "API-ERROR.json.lz4",
                {
                    "title": "bad-request",
                    "errors": [{"message": "a message from Twitter"}],
                },
                {
                    "cjw:apiEndpoint": "2/tweets/search/recent",
                    "cjw:apiParams": "expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
                    "cjw:httpStatus": "400",
                },
            )
        ]
    ) as tar_path:
        _assert_render(
            twitter.FetchResult(tar_path, []),
            None,
            [
                i18n_message(
                    "error.genericApiErrorV2",
                    {"title": "bad-request", "message": "a message from Twitter"},
                )
            ],
        )
