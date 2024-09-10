from urllib.parse import urlparse


def parse_url(url):
    """
    Parses a URL and returns the scheme, host, and port.

    Args:
        url (str): The URL to be parsed. It should follow the format 'scheme://host:port'.

    Returns:
        tuple: A tuple containing the scheme (str), host (str), and port (int).
               If the URL doesn't specify a port, returns None for the port.

    Raises:
        ValueError: If the URL is malformed or does not contain essential parts like scheme and host.

    Example:
        >>> parse_url("http://elasticsearch:9200")
        ('http', 'elasticsearch', 9200)

        >>> parse_url("https://example.com")
        ('https', 'example.com', None)
    """
    parsed = urlparse(url)

    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid URL: {url}")

    scheme = parsed.scheme
    host = parsed.hostname
    port = parsed.port

    return scheme, host, port
