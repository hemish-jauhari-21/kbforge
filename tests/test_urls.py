from kbforge.urls import host_matches, normalize_url


def test_lowercases_scheme_and_host():
    assert normalize_url("HTTP://Example.COM/Path") == "http://example.com/Path"


def test_strips_fragment():
    assert normalize_url("http://kb.test/a#section") == "http://kb.test/a"


def test_drops_tracking_params_and_sorts_rest():
    got = normalize_url("http://kb.test/a?b=2&a=1&utm_source=nl&fbclid=xyz")
    assert got == "http://kb.test/a?a=1&b=2"


def test_empty_path_becomes_slash():
    assert normalize_url("http://kb.test") == "http://kb.test/"


def test_resolves_relative_links_against_base():
    assert normalize_url("../other", base="http://kb.test/a/b/") == "http://kb.test/a/other"
    assert normalize_url("/top", base="http://kb.test/a/b/") == "http://kb.test/top"


def test_drops_default_ports():
    assert normalize_url("http://kb.test:80/a") == "http://kb.test/a"
    assert normalize_url("https://kb.test:443/a") == "https://kb.test/a"


def test_keeps_non_default_ports():
    assert normalize_url("http://kb.test:8080/a") == "http://kb.test:8080/a"


def test_rejects_non_http_schemes():
    assert normalize_url("mailto:a@b.c") is None
    assert normalize_url("javascript:void(0)") is None
    assert normalize_url("#") is None
    assert normalize_url("") is None


def test_rejects_invalid_ports_without_raising():
    assert normalize_url("http://example.com:bad/a") is None
    assert normalize_url("http://example.com:99999/a") is None
    assert normalize_url("http://example.com:-1/a") is None


def test_rejects_malformed_hosts_without_raising():
    assert normalize_url("http://[::1/a") is None
    assert normalize_url("http:///missing-host") is None
    assert normalize_url("not a url") is None


def test_host_matches_exact_and_subdomain():
    assert host_matches("http://kb.test/a", ["kb.test"])
    assert host_matches("http://docs.kb.test/a", ["kb.test"])
    assert not host_matches("http://evil-kb.test/a", ["kb.test"])
    assert not host_matches("http://other.test/a", ["kb.test"])


def test_host_matches_empty_list_allows_all():
    assert host_matches("http://anything.test/", [])
