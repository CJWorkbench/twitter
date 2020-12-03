import contextlib
import io
import tarfile
import tempfile
from pathlib import Path
from typing import ContextManager

import lz4.frame
import pyarrow as pa
import pyarrow.parquet
import pytest
from cjwmodule.testing.i18n import i18n_message
from pytest_httpx import HTTPXMock

import twitter

# When httpx.Client response with TimeoutError, that's because a request isn't
# mocked. It'll lead to an error, surely. To see the requests that didn't go
# through, write this before your test's failure:
#
#     print(repr(httpx_mock.get_requests()))


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


DefaultSecret = {
    "name": "x",
    "secret": {
        # After being processed by fetcher.secrets.prepare_secret()
        "consumer_key": "consumer-key",
        "consumer_secret": "consumer-secret",
        "resource_owner_key": "resource-owner-key",
        "resource_owner_secret": "resource-owner-secret",
    },
}


def P(
    querytype="user_timeline",
    username="username",
    query="query",
    listurl="listurl",
    accumulate=True,
):
    return dict(
        querytype=querytype,
        username=username,
        query=query,
        listurl=listurl,
        accumulate=accumulate,
    )


@contextlib.contextmanager
def _temp_parquet_file(table: pa.Table) -> ContextManager[Path]:
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        pa.parquet.write_table(table, path, version="2.0", compression="SNAPPY")
        yield path


def test_empty_query_and_secret():
    result = twitter.fetch_arrow(
        P(querytype="search", query=""),
        secrets={"twitter_credentials": None},
        last_fetch_result=None,
        input_table_parquet_path=None,
        output_path=Path("unused"),
    )
    assert not result.path.exists()
    assert not result.errors


def test_empty_query():
    result = twitter.fetch_arrow(
        P(querytype="search", query=""),
        secrets={"twitter_credentials": DefaultSecret},
        last_fetch_result=None,
        input_table_parquet_path=None,
        output_path=Path("unused"),
    )
    assert not result.path.exists()
    assert result.errors == [twitter.RenderError(i18n_message("error.noQuery"))]


def test_empty_secret():
    result = twitter.fetch_arrow(
        P(querytype="search", query="hi"),
        secrets={"twitter_credentials": None},
        last_fetch_result=None,
        input_table_parquet_path=None,
        output_path=Path("unused"),
    )
    assert not result.path.exists()
    assert result.errors == [twitter.RenderError(i18n_message("error.noCredentials"))]


def test_invalid_username():
    result = twitter.fetch_arrow(
        P(querytype="user_timeline", username="@@batman"),
        secrets={"twitter_credentials": DefaultSecret},
        last_fetch_result=None,
        input_table_parquet_path=None,
        output_path=Path("unused"),
    )
    assert not result.path.exists()
    assert result.errors == [twitter.RenderError(i18n_message("error.invalidUsername"))]


def test_invalid_list():
    result = twitter.fetch_arrow(
        P(querytype="lists_statuses", listurl="https://twitter.com/a/lists/@b"),
        secrets={"twitter_credentials": DefaultSecret},
        last_fetch_result=None,
        input_table_parquet_path=None,
        output_path=Path("unused"),
    )
    assert not result.path.exists()
    assert result.errors == [twitter.RenderError(i18n_message("error.invalidList"))]


def test_user_timeline_accumulate_atop_v1(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_1.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1263137241481859071&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_2.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:44 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1246131326031466497&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri Nov 27 2020 13:51:45 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        with tempfile.NamedTemporaryFile() as tarfile_tf:
            with tarfile.open(tarfile_tf.name, "w") as tar:
                # add a JSON file. Contents don't matter.
                ti = tarfile.TarInfo("656444340701634560.json.lz4")
                ti.size = len(b"unused contents")
                ti.pax_headers["cjw:apiEndpoint"] = "1.1/statuses/user_timeline.json"
                ti.pax_headers[
                    "cjw:apiParams"
                ] = "count=200&include_entities=false&screen_name=adamhooper&since_id=123&tweet_mode=extended"
                ti.pax_headers["cjw:httpStatus"] = "200"
                ti.pax_headers["cjw:nTweets"] = "1"
                tar.addfile(ti, io.BytesIO(b"unused contents"))

            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()

        page1 = next(parts)
        assert page1.name == "1275512525422018561.json.lz4"
        assert page1.api_endpoint == "1.1/statuses/user_timeline.json"
        assert (
            page1.api_params
            == "count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended"
        )
        assert page1.http_status == "200"
        assert page1.n_tweets == 2
        assert (
            lz4.frame.decompress(page1.body)
            == Path(
                "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_1.json"
            ).read_bytes()
        )

        page2 = next(parts)
        assert page2.name == "1246131326031466498.json.lz4"
        assert page2.n_tweets == 1

        oldpage = next(parts)
        assert oldpage.name == "656444340701634560.json.lz4"
        assert oldpage.body == b"unused contents"

        with pytest.raises(StopIteration):
            next(parts)


def test_user_timeline_accumulate_one_error_aborts_all(httpx_mock: HTTPXMock):
    # If Twitter gives 20 good responses and 1 error, ignore the good responses
    # and _only_ store the error. (Otherwise, there would be a gap in the
    # tarfile where we _should_ have fetched but we won't know we need to.)
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_1.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1263137241481859071&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"XXratelimitXX",
        status_code=429,
        headers={"date": "Fri Nov 27 2020 13:51:44 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        with tempfile.NamedTemporaryFile() as tarfile_tf:
            with tarfile.open(tarfile_tf.name, "w") as tar:
                # add a JSON file. Contents don't matter.
                ti = tarfile.TarInfo("656444340701634560.json.lz4")
                ti.size = len(b"unused contents")
                ti.pax_headers["cjw:apiEndpoint"] = "1.1/statuses/user_timeline.json"
                ti.pax_headers[
                    "cjw:apiParams"
                ] = "count=200&include_entities=false&screen_name=adamhooper&since_id=123&tweet_mode=extended"
                ti.pax_headers["cjw:httpStatus"] = "200"
                ti.pax_headers["cjw:nTweets"] = "1"
                tar.addfile(ti, io.BytesIO(b"unused contents"))

            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()

        page1 = next(parts)
        assert page1.name == "API-ERROR.lz4"
        assert lz4.frame.decompress(page1.body) == b"XXratelimitXX"
        assert page1.n_tweets is None
        assert page1.http_status == "429"
        # api_params are the ones with the error
        assert (
            page1.api_params
            == "count=200&include_entities=false&max_id=1263137241481859071&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended"
        )

        oldpage = next(parts)
        assert oldpage.name == "656444340701634560.json.lz4"
        assert oldpage.body == b"unused contents"

        with pytest.raises(StopIteration):
            next(parts)


def test_user_timeline_accumulate_ignore_duplicate_error(httpx_mock: HTTPXMock):
    # If Twitter gives the same error twice, that's probably because the
    # user entered the wrong params (e.g., a 404). We want to return the
    # exact same file as the last fetch, so Workbench won't store a new
    # version.
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"same response as last time",
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
        status_code=429,
    )

    with tempfile.NamedTemporaryFile() as tf:
        with tempfile.NamedTemporaryFile() as tarfile_tf:
            with tarfile.open(tarfile_tf.name, "w") as tar:
                ti = tarfile.TarInfo("API-ERROR.lz4")
                api_error_bytes = lz4.frame.compress(b"same response as last time")
                ti.size = len(api_error_bytes)
                ti.mtime = 1234.0
                ti.pax_headers["cjw:apiEndpoint"] = "1.1/statuses/user_timeline.json"
                ti.pax_headers[
                    "cjw:apiParams"
                ] = "count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended"
                ti.pax_headers["cjw:httpStatus"] = "429"
                tar.addfile(ti, io.BytesIO(api_error_bytes))

                ti = tarfile.TarInfo("656444340701634560.json.lz4")
                ti.size = len(b"unused body")
                ti.pax_headers["cjw:apiEndpoint"] = "1.1/statuses/user_timeline.json"
                ti.pax_headers[
                    "cjw:apiParams"
                ] = "count=200&include_entities=false&screen_name=adamhooper&tweet_mode=extended"
                ti.pax_headers["cjw:httpStatus"] = "200"
                ti.pax_headers["cjw:nTweets"] = "1"
                tar.addfile(ti, io.BytesIO(b"unused body"))
            tar_bytes = Path(tarfile_tf.name).read_bytes()

            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []
        assert result.path.read_bytes() == tar_bytes


def test_user_timeline_accumulate_replace_different_error(httpx_mock: HTTPXMock):
    # If Twitter gives the same error twice, that's probably because the
    # user entered the wrong params (e.g., a 404). We want to return the
    # exact same file as the last fetch, so Workbench won't store a new
    # version.
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"new error",
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
        status_code=429,
    )

    with tempfile.NamedTemporaryFile() as tf:
        with tempfile.NamedTemporaryFile() as tarfile_tf:
            with tarfile.open(tarfile_tf.name, "w") as tar:
                ti = tarfile.TarInfo("API-ERROR.lz4")
                api_error_bytes = lz4.frame.compress(b"same response as last time")
                ti.size = len(api_error_bytes)
                ti.mtime = 1234.0
                ti.pax_headers["cjw:apiEndpoint"] = "1.1/statuses/user_timeline.json"
                ti.pax_headers[
                    "cjw:apiParams"
                ] = "count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended"
                ti.pax_headers["cjw:httpStatus"] = "429"
                tar.addfile(ti, io.BytesIO(api_error_bytes))

                ti = tarfile.TarInfo("656444340701634560.json.lz4")
                ti.size = len(b"unused body")
                ti.pax_headers["cjw:apiEndpoint"] = "1.1/statuses/user_timeline.json"
                ti.pax_headers[
                    "cjw:apiParams"
                ] = "count=200&include_entities=false&screen_name=adamhooper&tweet_mode=extended"
                ti.pax_headers["cjw:httpStatus"] = "200"
                ti.pax_headers["cjw:nTweets"] = "1"
                tar.addfile(ti, io.BytesIO(b"unused body"))

            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []
        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()

        page1 = next(parts)
        assert page1.name == "API-ERROR.lz4"
        assert lz4.frame.decompress(page1.body) == b"new error"
        assert page1.mtime == 1606485103.0
        assert page1.n_tweets is None
        assert page1.http_status == "429"
        # api_params are the ones with the error
        assert (
            page1.api_params
            == "count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended"
        )

        oldpage = next(parts)
        assert oldpage.name == "656444340701634560.json.lz4"
        assert oldpage.body == b"unused body"


def test_user_timeline_accumulate_zero_tweets_atop_v0(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        with _temp_parquet_file(pa.table({"id": [656444340701634560]})) as parquet_path:
            parquet_bytes = parquet_path.read_bytes()
            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(parquet_path),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []
        assert result.path.read_bytes() == parquet_bytes


def test_user_timeline_accumulate_error_atop_v0(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"XXratelimitXX",
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
        status_code=429,
    )

    with tempfile.NamedTemporaryFile() as tf:
        with _temp_parquet_file(pa.table({"id": [656444340701634560]})) as parquet_path:
            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(parquet_path),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []
        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()
        assert list(part.name for part in parts) == [
            "API-ERROR.lz4",
            "LEGACY.parquet",
        ]


def test_user_timeline_accumulate_atop_v0(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_1.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1263137241481859071&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_2.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:44 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1246131326031466497&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri Nov 27 2020 13:51:45 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        with _temp_parquet_file(pa.table({"id": [656444340701634560]})) as parquet_path:
            parquet_bytes = parquet_path.read_bytes()
            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(parquet_path),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()

        page1 = next(parts)
        assert page1.name == "1275512525422018561.json.lz4"
        assert page1.api_endpoint == "1.1/statuses/user_timeline.json"
        assert (
            page1.api_params
            == "count=200&include_entities=false&screen_name=adamhooper&since_id=656444340701634560&tweet_mode=extended"
        )
        assert page1.http_status == "200"
        assert page1.n_tweets == 2
        assert (
            lz4.frame.decompress(page1.body)
            == Path(
                "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_1.json"
            ).read_bytes()
        )

        page2 = next(parts)
        assert page2.name == "1246131326031466498.json.lz4"
        assert page2.n_tweets == 1

        legacy = next(parts)
        assert legacy.name == "LEGACY.parquet"
        assert legacy.body == parquet_bytes

        with pytest.raises(StopIteration):
            next(parts)


def test_user_timeline_accumulate_atop_empty_file(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_2.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:44 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1246131326031466497&screen_name=adamhooper&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri Nov 27 2020 13:51:45 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        with tempfile.NamedTemporaryFile() as parquet_tf:
            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=True),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(Path(parquet_tf.name)),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        assert [p.name for p in result_file.get_result_parts()] == [
            "1246131326031466498.json.lz4"
        ]


def test_user_timeline_no_accumulate(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_1.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1263137241481859071&screen_name=adamhooper&tweet_mode=extended",
        data=Path(
            "tests/files/1_1_statuses_user_timeline_json_adamhooper_page_2.json"
        ).read_bytes(),
        headers={"date": "Fri Nov 27 2020 13:51:44 GMT"},
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&max_id=1246131326031466497&screen_name=adamhooper&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri Nov 27 2020 13:51:45 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        with _temp_parquet_file(pa.table({"id": [656444340701634560]})) as parquet_path:
            # last table -- will be ignored
            result = twitter.fetch_arrow(
                P(querytype="user_timeline", username="@adamhooper", accumulate=False),
                secrets={"twitter_credentials": DefaultSecret},
                last_fetch_result=twitter.FetchResult(parquet_path),
                input_table_parquet_path=None,
                output_path=Path(tf.name),
            )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()

        page1 = next(parts)
        assert page1.name == "1275512525422018561.json.lz4"
        assert page1.api_endpoint == "1.1/statuses/user_timeline.json"
        assert (
            page1.api_params
            == "count=200&include_entities=false&screen_name=adamhooper&tweet_mode=extended"
        )

        page2 = next(parts)
        assert page2.name == "1246131326031466498.json.lz4"
        assert page2.n_tweets == 1

        with pytest.raises(StopIteration):
            next(parts)


def test_user_timeline_no_tweets(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=adamhooper&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri Nov 27 2020 13:51:43 GMT"},
    )

    with tempfile.NamedTemporaryFile() as tf:
        result = twitter.fetch_arrow(
            P(querytype="user_timeline", username="@adamhooper", accumulate=False),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        assert result.errors == []

        # [2020-11-27, adamhooper] the framework treats "empty file" specially.
        # So we return an empty _tarfile_: it has some bytes, though they're all
        # 0.
        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = result_file.get_result_parts()
        assert list(parts) == []


def test_user_timeline_404_user_does_not_exist(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=wrongnamdsfe&tweet_mode=extended",
        data=b'{"errors":[{"code":34,"message":"Sorry, that page does not exist."}]}',
        headers={"date": "Thu, 26 Nov 2020 16:23:27 GMT"},
        status_code=404,
    )

    with tempfile.NamedTemporaryFile() as tf:
        result = twitter.fetch_arrow(
            P(querytype="user_timeline", username="wrongnamdsfe", accumulate=False),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = list(result_file.get_result_parts())
        assert len(parts) == 1
        part1 = parts[0]
        assert part1.name == "API-ERROR.lz4"
        assert (
            lz4.frame.decompress(part1.body)
            == b'{"errors":[{"code":34,"message":"Sorry, that page does not exist."}]}'
        )
        assert part1.mtime == 1606407807.0
        assert part1.api_endpoint == "1.1/statuses/user_timeline.json"
        assert (
            part1.api_params
            == "count=200&include_entities=false&screen_name=wrongnamdsfe&tweet_mode=extended"
        )
        assert part1.http_status == "404"
        assert part1.n_tweets is None


def test_user_timeline_401_tweets_are_private(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/statuses/user_timeline.json?count=200&include_entities=false&screen_name=elizabeth1&tweet_mode=extended",
        data=rb'{"request":"\/1.1\/statuses\/user_timeline.json","error":"Not authorized."}',
        headers={"date": "Thu, 26 Nov 2020 16:23:27 GMT"},
        status_code=401,
    )

    with tempfile.NamedTemporaryFile() as tf:
        result = twitter.fetch_arrow(
            P(querytype="user_timeline", username="elizabeth1", accumulate=False),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        assert result.errors == []

        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        parts = list(result_file.get_result_parts())
        assert len(parts) == 1
        part1 = parts[0]
        assert part1.name == "API-ERROR.lz4"
        assert part1.http_status == "401"


def test_list_statuses_owner_screen_name_slug_url(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/lists/statuses.json?count=200&include_entities=false&owner_screen_name=winnydejong&slug=NICAR2020&tweet_mode=extended",
        data=Path("tests/files/1_1_lists_statuses_NICAR2020_page_1.json").read_bytes(),
        headers={"date": "Fri, 27 Nov 2020 15:06:40 GMT"},
        status_code=200,
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/lists/statuses.json?count=200&include_entities=false&max_id=1332339491818901503&owner_screen_name=winnydejong&slug=NICAR2020&tweet_mode=extended",
        data=Path("tests/files/1_1_lists_statuses_NICAR2020_page_2.json").read_bytes(),
        headers={"date": "Fri, 27 Nov 2020 15:08:17 GMT"},
        status_code=200,
    )
    httpx_mock.add_response(
        url="https://api.twitter.com/1.1/lists/statuses.json?count=200&include_entities=false&max_id=1332340429472325632&owner_screen_name=winnydejong&slug=NICAR2020&tweet_mode=extended",
        data=b"[]",
        headers={"date": "Fri, 27 Nov 2020 15:11:03 GMT"},
        status_code=200,
    )

    with tempfile.NamedTemporaryFile() as tf:
        result = twitter.fetch_arrow(
            P(
                querytype="lists_statuses",
                listurl="https://twitter.com/winnydejong/lists/NICAR2020",
            ),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        assert result.errors == []
        assert result.path == Path(tf.name)
        result_file = twitter.FetchResultFile(result.path)
        assert list(part.name for part in result_file.get_result_parts()) == [
            "1332339491818901504.json.lz4",
            "1332340452041912327.json.lz4",
        ]


def test_list_statuses_owner_screen_name_slug(httpx_mock: HTTPXMock):
    # Don't mock any URLs. Let it error out -- let's only test the API params.
    with tempfile.NamedTemporaryFile() as tf:
        twitter.fetch_arrow(
            P(querytype="lists_statuses", listurl="winnydejong/NICAR2020"),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        result_file = twitter.FetchResultFile(Path(tf.name))
        assert list(part.api_params for part in result_file.get_result_parts()) == [
            "count=200&include_entities=false&owner_screen_name=winnydejong&slug=NICAR2020&tweet_mode=extended",
        ]


def test_list_statuses_list_id_url(httpx_mock: HTTPXMock):
    # Don't mock any URLs. Let it error out -- let's only test the API params.
    with tempfile.NamedTemporaryFile() as tf:
        twitter.fetch_arrow(
            P(
                querytype="lists_statuses",
                listurl="https://twitter.com/i/lists/1232648120690912345",
            ),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        result_file = twitter.FetchResultFile(Path(tf.name))
        assert list(part.api_params for part in result_file.get_result_parts()) == [
            "count=200&include_entities=false&list_id=1232648120690912345&tweet_mode=extended",
        ]


def test_list_statuses_list_id(httpx_mock: HTTPXMock):
    # Don't mock any URLs. Let it error out -- let's only test the API params.
    with tempfile.NamedTemporaryFile() as tf:
        twitter.fetch_arrow(
            P(
                querytype="lists_statuses",
                listurl="1232648120690912345",
            ),
            secrets={"twitter_credentials": DefaultSecret},
            last_fetch_result=None,
            input_table_parquet_path=None,
            output_path=Path(tf.name),
        )
        result_file = twitter.FetchResultFile(Path(tf.name))
        assert list(part.api_params for part in result_file.get_result_parts()) == [
            "count=200&include_entities=false&list_id=1232648120690912345&tweet_mode=extended",
        ]


# def test_search(httpx_mock: HTTPXMock):
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_1.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&next_token=b26v89c19zqg8o3fosesr6z0c2eac8nalgxjenpv5qjy5&query=science&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         # We nixed the "next_token" from 2_tweets_search_recent_page_2.json.
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:29:05 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         result = twitter.fetch_arrow(
#             P(querytype="search", query="science", accumulate=False),
#             secrets={"twitter_credentials": DefaultSecret},
#             last_fetch_result=None,
#             input_table_parquet_path=None,
#             output_path=Path(tf.name),
#         )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         parts = result_file.get_result_parts()
#
#         page1 = next(parts)
#         assert page1.name == "1332344846833639425.json.lz4"
#         assert page1.api_endpoint == "2/tweets/search/recent"
#         assert page1.n_tweets == 10
#         assert (
#             page1.api_params
#             == "expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname"
#         )
#
#         page2 = next(parts)
#         assert page2.name == "1332344823462965250.json.lz4"
#         assert page2.n_tweets == 10
#
#         with pytest.raises(StopIteration):
#             next(parts)
#
#
# def test_search_empty_results(httpx_mock: HTTPXMock):
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=nobodywrotethistextever&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=b'{"meta":{"result_count":0}}',
#         headers={"date": "Thu, 26 Nov 2020 15:21:40 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         result = twitter.fetch_arrow(
#             P(querytype="search", query="nobodywrotethistextever", accumulate=False),
#             secrets={"twitter_credentials": DefaultSecret},
#             last_fetch_result=None,
#             input_table_parquet_path=None,
#             output_path=Path(tf.name),
#         )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert not list(result_file.get_result_parts())
#
#
# def test_search_accumulate_empty_results(httpx_mock: HTTPXMock):
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=nobodywrotethistextever&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=b'{"meta":{"result_count":0}}',
#         headers={"date": "Thu, 26 Nov 2020 15:21:40 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with tempfile.NamedTemporaryFile() as tarfile_tf:
#             with tarfile.open(tarfile_tf.name, "w"):
#                 pass  # just write an empty tarfile
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="nobodywrotethistextever", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert not list(result_file.get_result_parts())
#
#
# def test_search_accumulate_recover_after_bug_160258591(httpx_mock: HTTPXMock):
#     # https://www.pivotaltracker.com/story/show/160258591
#     # 'id', 'retweet_count' and 'favorite_count' had type=text after
#     # accumulating an empty table. In file format v0, that postprocessing
#     # happened during fetch -- so the _stored_ data has the wrong type. We
#     # need to support this forever.
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=656444340701634560&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with _temp_parquet_file(
#             pa.table({"id": ["656444340701634560"]})
#         ) as parquet_path:
#             parquet_bytes = parquet_path.read_bytes()
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="science", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(parquet_path),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(result_file.get_result_parts())[1].body == parquet_bytes
#
#
# def test_search_accumulate_delete_empty_parquet(httpx_mock: HTTPXMock):
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with _temp_parquet_file(twitter.ARROW_SCHEMA.empty_table()) as parquet_path:
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="science", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(parquet_path),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(part.name for part in result_file.get_result_parts()) == [
#             "1332344823462965250.json.lz4"
#         ]
#
#
# def test_search_accumulate_read_max_tweet_id_from_legacy_parquet(httpx_mock: HTTPXMock):
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=15&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with _temp_parquet_file(
#             pa.table({"id": pa.array([15, 14, 13], pa.int64())})
#         ) as parquet_path:
#             parquet_bytes = parquet_path.read_bytes()
#         with tempfile.NamedTemporaryFile() as tarfile_tf:
#             with tarfile.open(tarfile_tf.name, "w") as tar:
#                 ti = tarfile.TarInfo("API-ERROR.lz4")
#                 ti.size = len(b"unused contents")
#                 ti.pax_headers["cjw:apiEndpoint"] = "2/tweets/search/recent"
#                 ti.pax_headers[
#                     "cjw:apiParams"
#                 ] = "expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=15&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname"
#                 ti.pax_headers["cjw:httpStatus"] = "429"
#                 tar.addfile(ti, io.BytesIO(b"unused contents"))
#
#                 # add the Parquet file
#                 ti = tarfile.TarInfo("LEGACY.parquet")
#                 ti.size = len(parquet_bytes)
#                 tar.addfile(ti, io.BytesIO(parquet_bytes))
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="science", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(part.name for part in result_file.get_result_parts()) == [
#             "1332344823462965250.json.lz4",
#             "LEGACY.parquet",
#         ]
#
#
# @override_settings(TWITTER_MAX_ROWS_PER_TABLE=14)
# def test_search_accumulate_truncate_and_delete_legacy_v0(httpx_mock: HTTPXMock):
#     # each mock response has 10 results
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=15&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_1.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&next_token=b26v89c19zqg8o3fosesr6z0c2eac8nalgxjenpv5qjy5&query=science&since_id=15&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         # We nixed the "next_token" from 2_tweets_search_recent_page_2.json.
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:29:05 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with _temp_parquet_file(
#             pa.table(
#                 {
#                     "id": pa.array(
#                         [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
#                         pa.int64(),
#                     )
#                 }
#             )
#         ) as parquet_path:
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="science", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(parquet_path),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(part.name for part in result_file.get_result_parts()) == [
#             "1332344846833639425.json.lz4",
#             "1332344823462965250.json.lz4",
#         ]
#
#
# @override_settings(TWITTER_MAX_ROWS_PER_TABLE=14)
# def test_search_accumulate_truncate_and_delete_legacy_v0_in_tarfile(
#     httpx_mock: HTTPXMock,
# ):
#     # The mock response has 10 results
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=25&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with _temp_parquet_file(
#             pa.table(
#                 {
#                     "id": pa.array(
#                         [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
#                         pa.int64(),
#                     )
#                 }
#             ),
#         ) as parquet_path:
#             parquet_bytes = parquet_path.read_bytes()
#         with tempfile.NamedTemporaryFile() as tarfile_tf:
#             with tarfile.open(tarfile_tf.name, "w") as tar:
#                 # add a JSON file. Contents don't matter.
#                 ti = tarfile.TarInfo("25.json.lz4")
#                 ti.size = len(b"unused contents")
#                 ti.pax_headers["cjw:apiEndpoint"] = "2/tweets/search/recent"
#                 ti.pax_headers[
#                     "cjw:apiParams"
#                 ] = "expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=15&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname"
#                 ti.pax_headers["cjw:httpStatus"] = "200"
#                 ti.pax_headers["cjw:nTweets"] = "10"
#                 tar.addfile(ti, io.BytesIO(b"unused contents"))
#
#                 # add the Parquet file
#                 ti = tarfile.TarInfo("LEGACY.parquet")
#                 ti.size = len(parquet_bytes)
#                 tar.addfile(ti, io.BytesIO(parquet_bytes))
#
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="science", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(part.name for part in result_file.get_result_parts()) == [
#             "1332344823462965250.json.lz4",
#             "25.json.lz4",
#         ]
#
#
# @override_settings(TWITTER_MAX_ROWS_PER_TABLE=24)
# def test_search_accumulate_keep_legacy_v0_in_tarfile(
#     httpx_mock: HTTPXMock,
# ):
#     # The mock response has 10 results
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=25&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with _temp_parquet_file(
#             pa.table(
#                 {
#                     "id": pa.array(
#                         [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
#                         pa.int64(),
#                     )
#                 }
#             ),
#         ) as parquet_path:
#             parquet_bytes = parquet_path.read_bytes()
#         with tempfile.NamedTemporaryFile() as tarfile_tf:
#             with tarfile.open(tarfile_tf.name, "w") as tar:
#                 # add a JSON file. Contents don't matter.
#                 ti = tarfile.TarInfo("25.json.lz4")
#                 ti.size = len(b"unused contents")
#                 ti.pax_headers["cjw:apiEndpoint"] = "2/tweets/search/recent"
#                 ti.pax_headers[
#                     "cjw:apiParams"
#                 ] = "expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&since_id=15&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname"
#                 ti.pax_headers["cjw:httpStatus"] = "200"
#                 ti.pax_headers["cjw:nTweets"] = "10"
#                 tar.addfile(ti, io.BytesIO(b"unused contents"))
#
#                 # add the Parquet file
#                 ti = tarfile.TarInfo("LEGACY.parquet")
#                 ti.size = len(parquet_bytes)
#                 tar.addfile(ti, io.BytesIO(parquet_bytes))
#
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="science", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(part.name for part in result_file.get_result_parts()) == [
#             "1332344823462965250.json.lz4",
#             "25.json.lz4",
#             "LEGACY.parquet",
#         ]
#
#
# def test_search_accumulate_delete_results_when_params_are_different(httpx_mock):
#     httpx_mock.add_response(
#         url="https://api.twitter.com/2/tweets/search/recent?expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=notscience&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname",
#         data=Path("tests/files/2_tweets_search_recent_page_2.json").read_bytes(),
#         headers={"date": "Fri, 27 Nov 2020 15:25:50 GMT"},
#         status_code=200,
#     )
#
#     with tempfile.NamedTemporaryFile() as tf:
#         with tempfile.NamedTemporaryFile() as tarfile_tf:
#             with tarfile.open(tarfile_tf.name, "w") as tar:
#                 ti = tarfile.TarInfo("12345.json.lz4")
#                 ti.size = len(b"unused contents")
#                 ti.pax_headers["cjw:apiEndpoint"] = "2/tweets/search/recent"
#                 ti.pax_headers[
#                     "cjw:apiParams"
#                 ] = "expansions=author_id%2Cin_reply_to_user_id%2Creferenced_tweets.id.author_id&max_results=100&query=science&tweet.fields=id%2Ctext%2Cauthor_id%2Ccreated_at%2Cin_reply_to_user_id%2Cpublic_metrics%2Csource%2Clang%2Creferenced_tweets&user.fields=id%2Cdescription%2Cusername%2Cname"
#                 ti.pax_headers["cjw:httpStatus"] = "200"
#                 tar.addfile(ti, io.BytesIO(b"unused contents"))
#
#             result = twitter.fetch_arrow(
#                 P(querytype="search", query="notscience", accumulate=True),
#                 secrets={"twitter_credentials": DefaultSecret},
#                 last_fetch_result=twitter.FetchResult(Path(tarfile_tf.name)),
#                 input_table_parquet_path=None,
#                 output_path=Path(tf.name),
#             )
#         assert result.errors == []
#         result_file = twitter.FetchResultFile(result.path)
#         assert list(part.name for part in result_file.get_result_parts()) == [
#             "1332344823462965250.json.lz4",
#         ]
