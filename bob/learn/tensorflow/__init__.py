def get_config():
    """
    Returns a string containing the configuration information.
    """
    import bob.extension
    return bob.extension.get_config(__name__)
