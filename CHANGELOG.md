2020-11-30
----------

* "Accumulate tweets": when the user changes the search, start from
  scratch. (Previously, a new search would ignore tweets older than the
  old results' newest tweet.)
* "Accumulate tweets": omit "retweet_count" and "favorite_count" columns.
  Each fetch's results comes from a different moment in time, meaning the
  values aren't comparable to one another. The numbers are thus meaningless,
  but users would try to extract meaning from them.
* "Accumulate tweets": upon network or Twitter error, do not delete
  accumulated tweets. When the user retries (manually or automatically),
  clear the error and continue accumulating.
* Use less CPU and memory during fetch (at the expense of more CPU and
  memory during render, which is where we want it).
* When not using "Accumulate tweets", every request will produce a new
  version. That's because Twitter's API responses contain small differences,
  even when the final table looks the same. (Previously we stored the final
  table. Now we store the API responses.)
  syntax errors.
* Use `httpx` for fetches. This gives nicer error messages for network
  errors.
* Allow translation (i18n) of errors.

There was also an ill-fated experiment with Twitter API v2. This was deployed
from 2020-11-30T17:00:00Z to 2020-11-30T17:30:00Z. During this period, some
"Search" queries produced errors. Users may re-run the queries to remove the
errors.
