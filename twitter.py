"""Twitter data loader: each row is a tweet.

FILE FORMAT (v1)
================

Either an empty file, or a tarfile with zero or more of the following files (in
order):

* Potentially one of the following error files:
    * `API-ERROR.lz4`: Similar to `[max-tweet-id].json.lz4`; but the body is an
       error (usually JSON, but not when `cjw:httpStatus` is `429`) and there is
       no `cjw:nTweets`.
    * `NETWORK-ERROR.json.lz4`: Similar to `[max-tweet-id].json.lz4`;
      but there is no `cjw.httpStatus` or `cjw.nTweets` and the JSON body is an
      `I18nMessage` (JSON Object with keys `id`, `arguments` and `source`).
* Any number of `[max-tweet-id].json.lz4` files (ending in ".json"), ordered by
  descending max-tweet-id. For each, tar extended headers apply:
    * mtime is HTTP `date` header from Twitter HTTP response
    * (pax) cjw:apiEndpoint: "1.1/statuses/user_timeline.json" or
                             "1.1/lists/statuses.json" or
                             "1.1/search/tweets.json"
    * (pax) cjw:apiParams: params, application/x-www-form-urlencoded
    * (pax) cjw:httpStatus: HTTP status code from Twitter
    * (pax) cjw:nTweets: Number of tweets (never 0)
    * Body is lz4-compressed Twitter API response (which is UTF8-encoded JSON).
* Potentially `LEGACY.parquet`: see "LEGACY FILE FORMAT (v0)" below.

Only one `API-ERROR.lz4` or `NETWORK-ERROR.json.lz4` file may exist. It
indicates that the most recent request did not produce new data. (We don't store
any indicator that other requests failed.)

If `LEGACY.parquet` exists, it is not empty.

This tarfile is designed for (optional) "accumulate". New fetches prepend files.
When TWITTER_MAX_ROWS_PER_TABLE is exceeded, older files are deleted from the
end.

Rationale:

    * Descending order: we don't need to sort tweets: Twitter returns them
      already-ordered.
    * Tar: we always read in order, and tar is minimal.
    * Uncompressed tar: we compress each file, not the archive, so "accumulate"
      won't need to decompress and recompress all contents. (That would be CPU-
      expensive.) Plus, there's no need to compress legacy Parquet files: they
      are already Snappy-compressed.
    * LZ4: it's fast. If we chose LZMA, a render's cost would mainly be in
      decompression.

LEGACY FILE FORMAT (v0)
-----------------------

Parquet file sorted by descending `id`, with columns:

* screen_name: utf8
* created_at: timestamp
* text: utf8
* retweet_count: int64 [1][2]
* favorite_count: int64 [1][2]
* in_reply_to_screen_name: utf8
* retweeted_status_screen_name: utf8
* user_description: utf8
* source: utf8
* lang: utf8
* id: int64 [2]

[1] We do not store the date each tweet was retrieved. So when users fetch
    with accumulate=True, "retweet_count" and "favorite_count" are arbitrary
    values: each was true at _some_ point in time, but one tweet's values
    aren't comparable to another tweet's values.
[2] Because of old bugs, we must support int64 stored as text.

RULES OF FETCHING
=================

* Any request that changes no outcome should not cause any file changes:
    * Zero tweets fetched? Don't modify anything (unless there was en error
      before -- in which case, delete it)
    * Same error as last fetch? Don't modify anything.
* Every `fetch()` call succeeds.

RULES OF "ACCUMULATE"
=====================

* Sorry -- we don't record any "gaps" yet.
* Circular buffer: when TWITTER_MAX_ROWS_PER_TABLE is exceeded, the oldest
  files are removed -- LEGACY.parquet if applicable, and JSON otherwise. The
  remaining number of tweets may exceed TWITTER_MAX_ROWS_PER_TABLE, but it won't
  exceed TWITTER_MAX_ROWS_PER_TABLE + 99. (100 is the maximum number of tweets
  per API request.)
* When parameters change, we reset. (We need to: otherwise, max_tweet_id
  interferes.) The exception: if we only have `LEGACY.parquet`, we can't know
  its parameters; so we assume it uses the new parameters.

RULES OF PAGINATED FETCHES
==========================

* Paginated fetches are "atomic". Either they all succeed (and lots of JSON
  files are added to the tarfile) or they all fail (and one error JSON is
  added).
* To stay under our time limit, we don't retry much.
"""

from __future__ import annotations

TWITTER_MAX_ROWS_PER_TABLE = 100_000

import asyncio
import datetime
import io
import json
import re
import shutil
import tarfile
import urllib.parse
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import httpx
import lz4.frame
import pyarrow as pa
import pyarrow.parquet
from cjwmodule import i18n
from cjwmodule.http import HttpError
from cjwmodule.http.client import download
from cjwmodule.http.errors import HttpErrorNotSuccess
from cjwparquet import file_has_parquet_magic_number
from oauthlib import oauth1
from oauthlib.common import urlencode


class RenderError(NamedTuple):
    """Mimics cjworkbench.cjwkernel.types.RenderError

    TODO move this to cjwmodule, so we can reuse it.
    """

    message: i18n.I18nMessage
    quick_fixes: List[None] = []


class FetchResult(NamedTuple):
    """Mimics cjworkbench.cjwkernel.types.FetchResult

    TODO move this to cjwmodule, so we can reuse it.
    """

    path: Path
    errors: List[RenderError] = []


def _migrate_params_v0_to_v1(params):
    """v0: 'querytype' is 0, 1 or 2. v1: 'user_timeline', 'search', 'lists_statuses'."""
    return {
        **params,
        "querytype": ["user_timeline", "search", "lists_statuses"][params["querytype"]],
    }


def migrate_params(params):
    if isinstance(params["querytype"], int):
        params = _migrate_params_v0_to_v1(params)

    return params


HTML_TAG_RE = re.compile("<[^>]*>")


def _parse_source(source: str) -> str:
    """Parse a Twitter Status 'source', to remove HTML tag."""
    return HTML_TAG_RE.sub("", source)


def _parse_lang(lang: str) -> Optional[str]:
    """Parse a Twitter Status 'lang', to set 'und' to None."""
    return None if lang == "und" else lang


def _parse_v1_tweet_text(tweet_json: Dict[str, Any]) -> str:
    if "retweeted_status" in tweet_json:
        status = tweet_json["retweeted_status"]
        screen_name = status["user"]["screen_name"]
        try:
            text = status["full_text"]
        except KeyError:
            # Some retweeted statuses don't have full_text.
            text = status["text"]
        return "RT @%s: %s" % (screen_name, text)
    else:
        return tweet_json["full_text"]


class Column(NamedTuple):
    name: str
    v1_path: List[Union[str, int, Callable]]
    dtype: pa.DataType
    accumulable: bool


Columns = [
    Column(
        "screen_name",
        ["user", "screen_name"],
        pa.utf8(),
        True,
    ),
    Column(
        "created_at",
        ["created_at", parsedate_to_datetime],
        pa.timestamp("ns"),
        True,
    ),
    Column("text", [_parse_v1_tweet_text], pa.utf8(), True),
    Column(
        "retweet_count",
        ["retweet_count"],
        pa.int64(),
        False,
    ),
    Column(
        "favorite_count",
        ["favorite_count"],
        pa.int64(),
        False,
    ),
    Column(
        "in_reply_to_screen_name",
        ["in_reply_to_screen_name"],
        pa.utf8(),
        True,
    ),
    Column(
        "retweeted_status_screen_name",
        ["retweeted_status", "user", "screen_name"],
        pa.utf8(),
        True,
    ),
    Column(
        "user_description",
        ["user", "description"],
        pa.utf8(),
        True,
    ),
    Column("source", ["source", _parse_source], pa.utf8(), True),
    Column("lang", ["lang", _parse_lang], pa.utf8(), True),
    Column("id", ["id"], pa.int64(), True),
]


ARROW_SCHEMA = pa.schema({column.name: column.dtype for column in Columns})
ACCUMULATED_SCHEMA = pa.schema(
    {column.name: column.dtype for column in Columns if column.accumulable}
)


def _recover_from_160258591(table: pa.Table) -> pa.Table:
    """Reset types of columns, in-place."""
    if table.schema == ARROW_SCHEMA:
        return table
    else:
        # https://www.pivotaltracker.com/story/show/160258591
        return pa.table(
            {
                column.name: table[column.name].cast(column.dtype)
                if column.name in table.column_names
                else pa.nulls(len(table), column.dtype)
                for column in Columns
            }
        )


def _tweets_to_column_list(
    tweets: List[Dict[str, Any]],
    path: List[Union[str, int, Callable]],
    expanded_tweets: Dict[str, Dict[str, str]],
    expanded_users: Dict[str, Dict[str, str]],
) -> List[Any]:
    records = tweets
    for path_part in path:
        if callable(path_part):
            records = [path_part(o) if o else None for o in records]
        elif path_part == "[type=retweeted]":
            records = [
                o[0] if o and o[0].get("type") == "retweeted" else None for o in records
            ]
        elif path_part == "[tweet]":
            records = [expanded_tweets.get(o) for o in records]  # .get(None) is okay
        elif path_part == "[user]":
            records = [expanded_users.get(o) for o in records]  # .get(None) is okay
        else:
            records = [None if o is None else o.get(path_part) for o in records]
    return records


def _twitter_v1_records_to_record_batch(
    data: Dict[str, Any], accumulated: bool
) -> pa.RecordBatch:
    if isinstance(data, dict) and "statuses" in data:
        data = data["statuses"]  # 1.1/search/tweets.json

    # `data` is an Array of tweets
    return pa.record_batch(
        [
            _tweets_to_column_list(
                data,
                column.v1_path,
                expanded_users={},
                expanded_tweets={},
            )
            for column in Columns
            if column.accumulable or not accumulated
        ],
        ACCUMULATED_SCHEMA if accumulated else ARROW_SCHEMA,
    )


# Inspired by https://github.com/twitter/twitter-text
OWNER_SCREEN_NAME_REGEX_PART = r"@?([a-zA-Z0-9_]{1,15})"
LIST_REGEX_PART = r"([a-zA-Z][-_a-zA-Z0-9]{0,24})"

OWNER_SCREEN_NAME_REGEX = re.compile(f"^{OWNER_SCREEN_NAME_REGEX_PART}$")
LIST_ID_URL_REGEX = re.compile("^(?:https?://)?twitter.com/i/lists/(\d+)$")
LIST_ID_REGEX = re.compile("^(\d+)$")
LIST_OWNER_SCREEN_NAME_SLUG_URL_REGEX = re.compile(
    f"^(?:https?://)?twitter.com/{OWNER_SCREEN_NAME_REGEX_PART}"
    f"/lists/{LIST_REGEX_PART}$"
)
LIST_OWNER_SCREEN_NAME_SLUG_REGEX = re.compile(
    f"^{OWNER_SCREEN_NAME_REGEX_PART}/{LIST_REGEX_PART}$"
)


def _render_legacy_file_format_v0_record_batches(
    parquet_bytes: bytes, accumulated: bool
) -> List[pa.RecordBatch]:
    arrow_table = pyarrow.parquet.read_table(io.BytesIO(parquet_bytes))

    # Clear Parquet metadata from table schema
    arrow_table = pa.Table.from_batches(
        arrow_table.to_batches(),
        schema=pa.schema(
            [
                arrow_table.schema.field(i).remove_metadata()
                for i in range(len(arrow_table.schema.names))
            ]
        ),
    )

    # We're guaranteed to have rows: otherwise,
    # FetchResultFile.get_result_parts() would not have yielded this Parquet
    # file
    arrow_table = _recover_from_160258591(arrow_table)

    if accumulated:
        arrow_table = arrow_table.select([c.name for c in Columns if c.accumulable])

    return arrow_table.to_batches()


def _render_api_error(
    api_endpoint: str, api_params: str, http_status: str, data: bytes
) -> i18n.I18nMessage:
    if http_status == "429":
        return i18n.trans(
            "error.tooManyRequests",
            "Twitter API rate limit exceeded. Please wait a few minutes and try again.",
        )

    if api_endpoint == "1.1/statuses/user_timeline":
        username = urllib.parse.parse_qs(api_params)["screen_name"][0]
        if http_status == "404":
            return i18n.trans(
                "error.userDoesNotExist",
                "User {username} does not exist",
                {"username": username},
            )
        elif http_status == "401":
            return i18n.trans(
                "error.userTweetsArePrivate",
                "User {username}'s tweets are private",
                {"username": username},
            )

    if api_endpoint.startswith("1.1/"):
        try:
            error = json.loads(data.decode("utf-8"))
            message = error["error"]
        except (KeyError, IndexError, ValueError):
            message = data.decode("utf-8")
        return i18n.trans(
            "error.genericApiErrorV1_1",
            "Error from Twitter API: {httpStatus} {error}",
            {"httpStatus": http_status, "error": message},
        )
    else:
        try:
            error = json.loads(data.decode("utf-8"))
            message = error["errors"][0]["message"]
        except (KeyError, IndexError, ValueError):
            message = data.decode("utf-8")
        return i18n.trans(
            "error.genericApiErrorV2",
            "Error from Twitter API: {title}: {message}",
            {"title": error["title"], "message": message},
        )


def _render_file_format_v1(
    path: Path,
    accumulated: bool,
) -> Tuple[Optional[pa.Table], List[i18n.I18nMessage]]:
    record_batches = []
    if path.stat().st_size > 0:
        result = FetchResultFile(path)
        for result_part in result.get_result_parts():
            if result_part.name == "API-ERROR.lz4":
                return None, [
                    _render_api_error(
                        result_part.api_endpoint,
                        result_part.api_params,
                        result_part.http_status,
                        lz4.frame.decompress(result_part.body),
                    )
                ]
            elif result_part.name == "NETWORK-ERROR.json.lz4":
                # NETWORK-ERROR is already an I18nMessage
                return None, [
                    i18n.I18nMessage(
                        **json.loads(lz4.frame.decompress(result_part.body)),
                    )
                ]
            elif result_part.name.endswith(".json.lz4"):
                json_records = json.loads(
                    lz4.frame.decompress(result_part.body).decode("utf-8")
                )
                record_batch = _twitter_v1_records_to_record_batch(
                    json_records, accumulated
                )
                if len(record_batch):
                    record_batches.append(record_batch)
            elif result_part.name == "LEGACY.parquet":
                record_batches.extend(
                    _render_legacy_file_format_v0_record_batches(
                        result_part.body, accumulated
                    )
                )
            else:
                raise NotImplementedError(
                    "Unhandled file '%s'" % result_part.name
                )  # pragma: no cover

    table = pa.Table.from_batches(
        record_batches, schema=ACCUMULATED_SCHEMA if accumulated else ARROW_SCHEMA
    )
    if len(table) > TWITTER_MAX_ROWS_PER_TABLE:
        table = table.slice(0, TWITTER_MAX_ROWS_PER_TABLE)

    table = table.combine_chunks()  # Workbench needs max 1 record batch
    return table, []


# Render just returns previously retrieved tweets
def render(arrow_table, params, output_path, *, fetch_result, **kwargs):
    if fetch_result is None:
        return []

    if fetch_result.errors:
        return list(error.message for error in fetch_result.errors)

    table, errors = _render_file_format_v1(
        fetch_result.path, accumulated=params["accumulate"]
    )
    if table is not None:
        with pa.ipc.RecordBatchFileWriter(output_path, table.schema) as writer:
            writer.write_table(table)

    return errors  # TODO format "id" column


def _parse_query(
    querytype: Literal["user_timeline", "search", "lists_statuses"], query: str
) -> Tuple[
    Literal[
        "1.1/statuses/user_timeline.json",
        "1.1/lists/statuses.json",
        "1.1/search/tweets.json",
    ],
    Dict[str, str],
]:
    """Parse params and return an endpoint + params.

    Ref: https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-rule

    The params don't include "common" query elements such as `tweet.fields`. We
    add those later.
    """

    if querytype == "lists_statuses":
        if m := LIST_OWNER_SCREEN_NAME_SLUG_URL_REGEX.match(query):
            params = dict(owner_screen_name=m.group(1), slug=m.group(2))
        elif m := LIST_OWNER_SCREEN_NAME_SLUG_REGEX.match(query):
            params = dict(owner_screen_name=m.group(1), slug=m.group(2))
        elif m := LIST_ID_URL_REGEX.match(query):
            params = {"list_id": m.group(1)}
        elif m := LIST_ID_REGEX.match(query):
            params = {"list_id": m.group(1)}
        else:
            raise ValueError(
                i18n.trans("error.invalidList", "Please enter a valid Twitter list URL")
            )
        return "1.1/lists/statuses.json", params
    elif querytype == "search":
        return "1.1/search/tweets.json", {"q": query}
    elif querytype == "user_timeline":
        if m := OWNER_SCREEN_NAME_REGEX.match(query):
            return "1.1/statuses/user_timeline.json", {"screen_name": m.group(1)}
        else:
            raise ValueError(
                i18n.trans(
                    "error.invalidUsername", "Please enter a valid Twitter username"
                )
            )
    else:
        raise NotImplementedError(
            "Invalid querytype '%s'" % querytype
        )  # pragma: no cover


def _parquet_file_to_max_tweet_id(f: Union[Path, BinaryIO]) -> Optional[int]:
    """Given a Path or BinaryIO Parquet file, return max tweet ID.

    Return None if there are no rows.

    Reads ID column into memory. 100,000 IDs @ 8b/ID => 1MB RAM.
    """
    try:
        ids_table = pa.parquet.read_table(f, columns=["id"], use_threads=False)
    except pa.ArrowInvalid:
        # There's no "id" column ... or the entire Parquet file is trash
        return None
    if not len(ids_table):
        return None

    ids = ids_table["id"]
    # int() because https://www.pivotaltracker.com/story/show/160258591
    return int(ids[0].as_py())


def _utf8_json_encode(obj: Any) -> bytes:
    return json.dumps(
        obj,
        ensure_ascii=True,
        check_circular=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class ResultPart(NamedTuple):
    """File stored in FetchResultFile, holding tweets and/or metadata."""

    name: str
    """Name of file:

    * API-ERROR.lz4 (must be first)
    * NETWORK-ERROR.json.lz4 (must be first)
    * [max-tweet-id].json.lz4
    * LEGACY.parquet (must be last)
    """

    body: bytes
    """Contents of the file.

    If this is LEGACY.parquet, this is the binary file data.

    If this is a .json.lz4 file, this is LZ4-compressed, UTF-8 encoded JSON.
    """

    mtime: float
    """UNIX timestamp of Twitter's response (or our error)."""

    api_endpoint: Optional[
        Literal[
            "1.1/lists/statuses.json",
            "1.1/statuses/user_timeline.json",
            "1.1/search/tweets.json",
        ]
    ] = None
    """Endpoint that led to this file's creation."""

    api_params: Optional[str] = None
    """Params (application/x-www-form-urlencoded) that led to this file's creation."""

    http_status: Optional[str] = None
    """HTTP response from Twitter (if there was one)."""

    n_tweets: Optional[int] = None
    """Number of tweets in this file (for [max-tweet-id].json.lz4 only)."""

    @classmethod
    def for_parquet_bytes(cls, body: bytes) -> ResultPart:
        return ResultPart(name="LEGACY.parquet", mtime=0.0, body=body)

    @classmethod
    def for_tarinfo(cls, ti: tarfile.TarInfo, body: bytes) -> ResultPart:
        return ResultPart(
            name=ti.name,
            mtime=ti.mtime,
            api_endpoint=ti.pax_headers.get("cjw:apiEndpoint"),
            api_params=ti.pax_headers.get("cjw:apiParams"),
            http_status=ti.pax_headers.get("cjw:httpStatus"),
            n_tweets=int(ti.pax_headers.get("cjw:nTweets", "0")) or None,  # never 0
            body=body,
        )


class FetchResultFile:
    """Low-memory representation of last_fetch_result.

    The goal is to keep RAM and CPU usage low, so fetches can be faster.
    """

    def __init__(self, path: Optional[Path]):
        self.path = path

    def get_accumulatable_api_endpoint_and_params(self) -> Optional[Dict[str, str]]:
        """Return (endpoint, params) that generated this FetchResultFile.

        Return None if we cannot know or if there never was one.
        """
        if file_has_parquet_magic_number(self.path):
            return None

        with tarfile.open(self.path, mode="r") as tf:
            ti = tf.firstmember
            if ti is None:
                return None

            # This can _only_ be a ERROR.json.lz4 or [id].json.lz4 file.
            # A tarfile is either empty, or it contains a file _before_
            # LEGACY.parquet.
            api_endpoint = ti.pax_headers["cjw:apiEndpoint"]
            api_params = ti.pax_headers["cjw:apiParams"]
            return api_endpoint, {
                key: values[0]
                for key, values in urllib.parse.parse_qs(api_params).items()
                if key
                not in {
                    "expansions",
                    "tweet.fields",
                    "user.fields",
                    "max_results",
                    "count",
                    "next_token",
                    "since_id",
                    "include_entities",
                    "tweet_mode",
                }
            }

    def get_max_tweet_id(self) -> Optional[int]:
        """Calculate the maximum tweet ID throughout the fetched file.

        Return None if no tweets have been fetched.
        """
        if file_has_parquet_magic_number(self.path):
            return _parquet_file_to_max_tweet_id(self.path)

        with tarfile.open(self.path, mode="r") as tf:
            for ti in tf:
                if ti.name.endswith(".json.lz4") and "ERROR" not in ti.name:
                    return int(ti.name[:-9])
                if ti.name == "LEGACY.parquet":
                    with tarfile.TarFile.fileobject(tf, ti) as f:
                        return _parquet_file_to_max_tweet_id(f)

        return None  # No tweets

    def get_result_parts(self) -> Iterable[ResultPart]:
        """Iterate over ResultParts. May be interrupted.

        This is a generator so RAM usage is at a minimum. We only load one file
        into RAM at a time.
        """
        if file_has_parquet_magic_number(self.path):
            # If the file has no rows, then don't yield it. This way, we'll
            # never write an empty LEGACY.parquet into a v1 file.
            is_empty = _parquet_file_to_max_tweet_id(self.path) is None
            if not is_empty:
                # Assume the Parquet file is fairly small, so it fits in RAM
                yield ResultPart.for_parquet_bytes(self.path.read_bytes())
            return

        with tarfile.open(self.path, "r") as tf:
            for ti in tf:
                yield ResultPart.for_tarinfo(
                    ti, tarfile.TarFile.fileobject(tf, ti).read()
                )

    def get_error_result_part(self) -> Optional[ResultPart]:
        """Return the last file, if it's error."""
        if file_has_parquet_magic_number(self.path):
            return None

        with tarfile.open(self.path, "r") as tf:
            ti = tf.firstmember
            if ti is not None and "ERROR" in ti.name:
                # Read the body into RAM. When we return, we'll close the tarfile
                # so any fileobject within it would be invalid.
                body = tarfile.TarFile.fileobject(tf, ti).read()
                return ResultPart.for_tarinfo(ti, body)
        return None


def _write_result_part_to_tarfile(tf: tarfile.TarFile, result: ResultPart) -> None:
    """Append bytes to `tf`."""
    info = tarfile.TarInfo(result.name)
    info.size = len(result.body)
    for attr_name, header_name in (
        ("mtime", "mtime"),
        ("api_endpoint", "cjw:apiEndpoint"),
        ("api_params", "cjw:apiParams"),
        ("http_status", "cjw:httpStatus"),
        ("n_tweets", "cjw:nTweets"),
    ):
        value = getattr(result, attr_name)
        if value is not None:
            info.pax_headers[header_name] = str(value)
    tf.addfile(info, io.BytesIO(result.body))


async def _fetch_paginated(
    endpoint: Literal[
        "1.1/lists/statuses.json",
        "1.1/statuses/user_timeline.json",
        "1.1/search/tweets.json",
    ],
    params: Dict[str, Any],
    *,
    credentials: Dict[str, str],
) -> List[ResultPart]:
    """Fetch pages of results from Twitter API.

    Return empty list if there are no tweets.

    Return a single "API-ERROR.lz4" result if Twitter responds negatively
    to any request.

    Return a single "NETWORK-ERROR.json.lz4" result if we fail to receive a
    Twitter response to any request.

    Otherwise, return tweets ordered from newest to oldest.
    """
    oauth_client = oauth1.Client(
        client_key=credentials["consumer_key"],
        client_secret=credentials["consumer_secret"],
        resource_owner_key=credentials["resource_owner_key"],
        resource_owner_secret=credentials["resource_owner_secret"],
    )
    async with httpx.AsyncClient() as client:
        if endpoint == "1.1/search/tweets.json":
            return await _fetch_paginated_1_1(
                client,
                endpoint,
                params,
                oauth_client=oauth_client,
                tweets_per_request=100,
                max_n_requests=10,
            )
        elif endpoint == "1.1/statuses/user_timeline.json":
            return await _fetch_paginated_1_1(
                client,
                endpoint,
                params,
                oauth_client=oauth_client,
                tweets_per_request=200,
                max_n_requests=16,  # 3,200 tweets: [2020-11-26] Twitter's maximum
            )
        elif endpoint == "1.1/lists/statuses.json":
            return await _fetch_paginated_1_1(
                client,
                endpoint,
                params,
                oauth_client=oauth_client,
                tweets_per_request=200,
                max_n_requests=5,  # arbitrary => 1,000 tweets
            )
        else:
            raise NotImplementedError(
                "Unknown endpoint '%s'" % endpoint
            )  # pragma: no cover


def _lz4_compress(b: bytes) -> bytes:
    return lz4.frame.compress(b, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC)


async def _call_twitter_api_once(
    client: httpx.Client,
    api_endpoint: str,
    params: Dict[str, str],
    *,
    oauth_client: oauth1.Client,
    get_n_tweets: Callable[[Dict[str, Any]], int],
    get_max_tweet_id: Callable[[Dict[str, Any]], int],
    get_next_token: Callable[[Dict[str, Any]], Union[int, Optional[str]]],
) -> Tuple[Optional[ResultPart], Optional[Union[int, str]]]:
    """Send a single request to the Twitter API; return (ResultPart, next_token).

    If the result has no tweets, return `None`.

    Otherwise, wrap the result in a `ResultPart`.

    Undefined behavior if `get_n_tweets()` raises an error, or if Twitter
    returns non-JSON.

    For Twitter API v1, "get_next_token" can return the minimum tweet ID.
    Callers can use that to construct a "max_id" for a subsequent request.
    """
    HEADERS = {"Accept": "application/json"}

    api_params = urlencode(list(sorted(params.items())))
    url = f"https://api.twitter.com/{api_endpoint}?{api_params}"
    url, headers, body = oauth_client.sign(url, headers=HEADERS)

    body = io.BytesIO()
    try:
        result = await download(url, body, headers=headers, httpx_client=client)
    except HttpErrorNotSuccess as err:
        return (
            ResultPart(
                "API-ERROR.lz4",
                mtime=parsedate_to_datetime(err.response.headers["date"]).timestamp(),
                api_endpoint=api_endpoint,
                api_params=api_params,
                http_status=str(err.response.status_code),
                body=_lz4_compress(body.getbuffer()),
            ),
            None,
        )
    except HttpError as err:
        return (
            ResultPart(
                "NETWORK-ERROR.json.lz4",
                mtime=datetime.datetime.utcnow().timestamp(),
                api_endpoint=api_endpoint,
                api_params=api_params,
                body=_lz4_compress(_utf8_json_encode(err.i18n_message._asdict())),
            ),
            None,
        )

    # missing HTTP date header? Undefined behavior
    date_header_value = next(
        value for key, value in result.headers if key.lower() == "date"
    )
    # not valid UTF-8 JSON? Undefined behavior
    data = json.loads(str(body.getvalue(), encoding="utf-8"))
    # missing JSON fields, or not Array response? Undefined behavior
    n_tweets = get_n_tweets(data)
    if n_tweets == 0:
        return None, None  # don't write this API response to the tarfile
    max_tweet_id = get_max_tweet_id(data)
    next_token = get_next_token(data)

    return (
        ResultPart(
            "%d.json.lz4" % max_tweet_id,
            # invalid date format? Undefined behavior
            mtime=parsedate_to_datetime(date_header_value).timestamp(),
            api_endpoint=api_endpoint,
            api_params=api_params,
            http_status=str(result.status_code),
            n_tweets=n_tweets,
            body=_lz4_compress(body.getbuffer()),
        ),
        next_token,
    )


async def _fetch_paginated_1_1(
    client: httpx.Client,
    endpoint: Literal[
        "1.1/lists/statuses.json",
        "1.1/statuses/user_timeline.json",
        "1.1/search/tweets.json",
    ],
    params: Dict[str, Any],
    *,
    oauth_client: oauth1.Client,
    tweets_per_request: int,
    max_n_requests: int,
) -> List[ResultPart]:
    """Fetch from Twitter API v1.1.

    Return empty list if there are no tweets.

    Return a single "API-ERROR.lz4" result if Twitter responds negatively
    to any request.

    Return a single "NETWORK-ERROR.json.lz4" result if we fail to receive a
    Twitter response to any request.

    Otherwise, return tweets ordered from newest to oldest.
    """
    retval: List[ResultPart] = []  # result, unless we hit an error
    max_id = None

    if endpoint == "1.1/search/tweets.json":

        def get_n_tweets(data):
            return len(data["statuses"])

        def get_max_tweet_id(data):
            return data["statuses"][0]["id"]

        def get_next_token(data):
            return data["statuses"][-1]["id"]

    else:
        get_n_tweets = len

        def get_max_tweet_id(data):
            return data[0]["id"]

        def get_next_token(data):
            return data[-1]["id"]

    for _ in range(max_n_requests):
        page_params = {
            **params,
            "tweet_mode": "extended",
            "include_entities": "false",
            "count": str(tweets_per_request),
        }
        if max_id is not None:
            page_params["max_id"] = str(max_id)

        result_part, min_id = await _call_twitter_api_once(
            client,
            endpoint,
            page_params,
            oauth_client=oauth_client,
            get_n_tweets=get_n_tweets,
            get_max_tweet_id=get_max_tweet_id,
            get_next_token=get_next_token,  # next_token is v2 nomenclature
        )

        if not result_part:
            break  # no tweets: don't add this to the tarfile

        if "ERROR" in result_part.name:
            return [result_part]

        retval.append(result_part)

        # Now loop and request again ... using the new `max_id` to find only
        # older tweets
        max_id = min_id - 1

    return retval


def fetch_arrow(
    params: Dict[str, Any],
    secrets: Dict[str, Any],
    last_fetch_result: FetchResult,
    input_table_parquet_path: Optional[Path],
    output_path: Path,
    **kwargs,
) -> FetchResult:
    query: str = params[
        {"user_timeline": "username", "search": "query", "lists_statuses": "listurl"}[
            params["querytype"]
        ]
    ].strip()
    credentials = (secrets.get("twitter_credentials") or {}).get("secret")

    if not query and not credentials:
        return FetchResult(output_path)  # Don't create a version
    if not query:
        return FetchResult(
            output_path,
            [RenderError(i18n.trans("error.noQuery", "Please enter a query"))],
        )
    if not credentials:
        return FetchResult(
            output_path,
            [
                RenderError(
                    i18n.trans("error.noCredentials", "Please sign in to Twitter")
                )
            ],
        )
    try:
        api_endpoint, api_params = _parse_query(params["querytype"], query)
    except ValueError as err:
        return FetchResult(output_path, [RenderError(err.args[0])])  # an I18nMessage

    if (
        params["accumulate"]
        and last_fetch_result
        and last_fetch_result.path.stat().st_size > 0
    ):
        last_result = FetchResultFile(last_fetch_result.path)
        last_endpoint_and_params = (
            last_result.get_accumulatable_api_endpoint_and_params()
        )
        if (
            last_endpoint_and_params is None  # accumulate atop a v0 legacy file
            or last_endpoint_and_params == (api_endpoint, api_params)  # accumulate
        ):
            since_id = last_result.get_max_tweet_id()  # may be None
            if since_id is not None:
                api_params["since_id"] = since_id
        else:
            # last_result had a different query from this one. Throw out the
            # old tweets and start anew.
            last_result = None
    else:
        last_result = None

    new_results = asyncio.run(
        _fetch_paginated(api_endpoint, api_params, credentials=credentials)
    )

    if last_result and not new_results:
        shutil.copyfile(last_result.path, output_path)
        return FetchResult(output_path, last_fetch_result.errors)

    if last_result and len(new_results) == 1 and "ERROR" in new_results[0].name:
        new_error = new_results[0]
        last_error = last_result.get_error_result_part()
        if (
            last_error
            and new_error.name == last_error.name
            and new_error.http_status == last_error.http_status
            and lz4.frame.decompress(new_error.body)
            == lz4.frame.decompress(last_error.body)
        ):
            # Aww, our error is the same as last time! This is one big no-op.
            shutil.copyfile(last_result.path, output_path)
            return FetchResult(output_path, last_fetch_result.errors)

    n_tweets_allowed = TWITTER_MAX_ROWS_PER_TABLE
    with tarfile.open(output_path, mode="w") as tf:
        for result_part in new_results:
            _write_result_part_to_tarfile(tf, result_part)
            if result_part.n_tweets is not None:
                n_tweets_allowed -= result_part.n_tweets
                if n_tweets_allowed <= 0:
                    break
        if last_result and n_tweets_allowed > 0:
            for result_part in last_result.get_result_parts():
                if "ERROR" in result_part.name:
                    # new_results either gave an error or gave no errors. Either
                    # way, don't write the _old_ error.
                    continue
                _write_result_part_to_tarfile(tf, result_part)
                if result_part.n_tweets is not None:
                    n_tweets_allowed -= result_part.n_tweets
                    if n_tweets_allowed <= 0:
                        break
    return FetchResult(output_path)
