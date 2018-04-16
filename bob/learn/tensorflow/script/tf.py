"""The main entry for bob tf (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points('bob.learn.tensorflow.cli'))
@click.group()
def tf():
    """Tensorflow-related commands."""
    pass
