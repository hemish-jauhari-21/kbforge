"""URL normalization used by the crawler.

Normalization is what makes crawl dedup work: the same page reachable as
``HTTP://Example.com/a?utm_source=x#frag`` and ``http://example.com/a``
must collapse to one canonical string. Rules:

- lowercase scheme and host
- strip ``www.`` prefix? No: ``www.example.com`` may be a different host,
  so the host is kept as-is (minus default ports)
- drop the fragment, drop non-http(s) schemes (mailto:, javascript:, ...)
- drop common tracking query parameters (utm_*, fbclid, gclid, ...),
  URL-decode, re-encode and sort the remaining parameters
- empty path becomes ``/``
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

TRACKING_PARAM_PREFIXES = ("utm_",)
TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "igshid",
    "ref_src",
    "ref_url",
    "yclid",
}


def is_tracking_param(name: str) -> bool:
    lowered = name.lower()
    return lowered in TRACKING_PARAMS or lowered.startswith(TRACKING_PARAM_PREFIXES)


def normalize_url(href: str, base: str | None = None) -> str | None:
    """Normalize ``href`` (resolved against ``base`` if given).

    Returns the canonical URL string, or ``None`` when the href is not a
    crawlable http(s) URL (empty, fragment-only, javascript:, mailto:...).
    """
    if href is None:
        return None
    href = href.strip()
    if not href:
        return None

    try:
        absolute = urljoin(base, href) if base else href
        parts = urlsplit(absolute)
        scheme = parts.scheme.lower()
        if scheme not in ("http", "https"):
            return None

        host = (parts.hostname or "").lower()
        if not host:
            return None
        # Accessing .port validates the port and raises ValueError for values
        # such as "http://example.com:bad/" or out-of-range ports.
        port = parts.port
    except ValueError:
        # urlsplit/urljoin can also raise for malformed IPv6 hosts and similar
        # invalid inputs. Malformed links must be skipped, never crash a crawl.
        return None

    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        host = f"{host}:{port}"

    path = parts.path or "/"

    kept_params = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not is_tracking_param(k)
    ]
    kept_params.sort()
    query = urlencode(kept_params, doseq=True)

    # Fragments never identify a distinct resource on the server side.
    return urlunsplit((scheme, host, path, query, ""))


def host_matches(url: str, allowed_domains: list[str]) -> bool:
    """True when ``url``'s host is in ``allowed_domains`` or a subdomain of one."""
    if not allowed_domains:
        return True
    host = (urlsplit(url).hostname or "").lower()
    for domain in allowed_domains:
        domain = domain.lower()
        if host == domain or host.endswith("." + domain):
            return True
    return False
