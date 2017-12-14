def get_from_config_or_commandline(config,
                                   keyword,
                                   args,
                                   defaults,
                                   default_is_valid=True):
    """Takes the value from command line, config file, and default value with
    this precedence.

    Only several command line options can be used with this function:
    - boolean flags
    - repeating flags (like --verbose)
    - options where the user will never provide the default value through
      command line. For example when [default: None]

    Parameters
    ----------
    config : :any:`module`
        The loaded config files.
    keyword : str
        The keyword to load from the config file or through command line.
    args : dict
        The arguments parsed with docopt.
    defaults : dict
        The arguments parsed with docopt when ``argv=[]``.
    default_is_valid : bool, optional
        If False, will raise an exception if the final parsed value is the
        default value.

    Returns
    -------
    object
        The bool or integer value of the corresponding keyword.

    Example
    -------
    >>> from bob.extension.config import load as read_config_file
    >>> defaults = docopt(docs, argv=[""])
    >>> args = docopt(docs, argv=argv)
    >>> config_files = args['<config_files>']
    >>> config = read_config_file(config_files)

    >>> verbosity = get_from_config_or_commandline(config, 'verbose', args,
    ...                                            defaults)

    """
    arg_keyword = '--' + keyword.replace('_', '-')

    # load from config first
    value = getattr(config, keyword, defaults[arg_keyword])

    # override it if provided by command line arguments
    if args[arg_keyword] != defaults[arg_keyword]:
        value = args[arg_keyword]

    if not default_is_valid and value == defaults[arg_keyword]:
        raise ValueError(
            "The value provided for {} is not valid.".format(keyword))

    return value
