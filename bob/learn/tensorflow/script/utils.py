import click


def eager_execution_option(**kwargs):
    """Adds an option to your command to enable eager execution of tensorflow

    Returns
    -------
     callable
      A decorator to be used for adding this option to click commands
    """
    def custom_eager_execution_option(func):
        def callback(ctx, param, value):
            if not value or ctx.resilient_parsing:
                return
            import tensorflow as tf
            tf.enable_eager_execution()
            if not tf.executing_eagerly():
                raise click.ClickException(
                    "Could not enable tensorflow eager execution mode!")
            else:
                click.echo("Executing tensorflow operations eagerly!")

        return click.option(
            '-e', '--eager', is_flag=True, callback=callback,
            expose_value=False, is_eager=True, **kwargs)(func)
    return custom_eager_execution_option
