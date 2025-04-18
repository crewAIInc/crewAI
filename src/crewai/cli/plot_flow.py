import importlib.util
import os
import sys

import click


def plot_flow() -> None:
    """
    Plot the flow by finding and importing the plot function from the project's main.py file.
    """
    main_file_path = os.path.join(os.getcwd(), "main.py")
    
    try:
        if os.path.exists(main_file_path):
            spec = importlib.util.spec_from_file_location("main", main_file_path)
            if spec is not None and spec.loader is not None:
                main = importlib.util.module_from_spec(spec)
                sys.modules["main"] = main
                spec.loader.exec_module(main)
                
                if hasattr(main, "plot"):
                    main.plot()
                else:
                    click.echo("Error: No plot function found in main.py", err=True)
            else:
                click.echo("Error: Could not load main.py module", err=True)
        else:
            click.echo("Error: Could not find main.py in the current directory", err=True)
            
    except Exception as e:
        click.echo(f"An error occurred while plotting the flow: {e}", err=True)
