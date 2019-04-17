"""The main entry for bob keras (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins
from bob.extension.scripts.click_helper import AliasedGroup
from .utils import eager_execution_option


@with_plugins(pkg_resources.iter_entry_points('bob.learn.tensorflow.keras_cli'))
@click.group(cls=AliasedGroup)
@eager_execution_option()
def keras():
    """Keras-related commands."""
    pass
