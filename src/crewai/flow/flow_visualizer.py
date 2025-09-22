# flow_visualizer.py

import os

from pyvis.network import Network  # type: ignore[import-untyped]

from crewai.flow.config import COLORS, NODE_STYLES
from crewai.flow.html_template_handler import HTMLTemplateHandler
from crewai.flow.legend_generator import generate_legend_items_html, get_legend_items
from crewai.flow.path_utils import safe_path_join
from crewai.flow.utils import calculate_node_levels
from crewai.flow.visualization_utils import (
    add_edges,
    add_nodes_to_network,
    compute_positions,
)


class FlowPlot:
    """Handles the creation and rendering of flow visualization diagrams."""

    def __init__(self, flow):
        """
        Initialize FlowPlot with a flow object.

        Parameters
        ----------
        flow : Flow
            A Flow instance to visualize.

        Raises
        ------
        ValueError
            If flow object is invalid or missing required attributes.
        """
        if not hasattr(flow, "_methods"):
            raise ValueError("Invalid flow object: missing '_methods' attribute")
        if not hasattr(flow, "_listeners"):
            raise ValueError("Invalid flow object: missing '_listeners' attribute")
        if not hasattr(flow, "_start_methods"):
            raise ValueError("Invalid flow object: missing '_start_methods' attribute")

        self.flow = flow
        self.colors = COLORS
        self.node_styles = NODE_STYLES

    def plot(self, filename):
        """
        Generate and save an HTML visualization of the flow.

        Parameters
        ----------
        filename : str
            Name of the output file (without extension).

        Raises
        ------
        ValueError
            If filename is invalid or network generation fails.
        IOError
            If file operations fail or visualization cannot be generated.
        RuntimeError
            If network visualization generation fails.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("Filename must be a non-empty string")

        try:
            # Initialize network
            net = Network(
                directed=True,
                height="750px",
                width="100%",
                bgcolor=self.colors["bg"],
                layout=None,
            )

            # Set options to disable physics
            net.set_options(
                """
                var options = {
                    "nodes": {
                        "font": {
                            "multi": "html"
                        }
                    },
                    "physics": {
                        "enabled": false
                    }
                }
            """
            )

            # Calculate levels for nodes
            try:
                node_levels = calculate_node_levels(self.flow)
            except Exception as e:
                raise ValueError(f"Failed to calculate node levels: {e!s}") from e

            # Compute positions
            try:
                node_positions = compute_positions(self.flow, node_levels)
            except Exception as e:
                raise ValueError(f"Failed to compute node positions: {e!s}") from e

            # Add nodes to the network
            try:
                add_nodes_to_network(net, self.flow, node_positions, self.node_styles)
            except Exception as e:
                raise RuntimeError(f"Failed to add nodes to network: {e!s}") from e

            # Add edges to the network
            try:
                add_edges(net, self.flow, node_positions, self.colors)
            except Exception as e:
                raise RuntimeError(f"Failed to add edges to network: {e!s}") from e

            # Generate HTML
            try:
                network_html = net.generate_html()
                final_html_content = self._generate_final_html(network_html)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to generate network visualization: {e!s}"
                ) from e

            # Save the final HTML content to the file
            try:
                with open(f"{filename}.html", "w", encoding="utf-8") as f:
                    f.write(final_html_content)
                print(f"Plot saved as {filename}.html")
            except IOError as e:
                raise IOError(
                    f"Failed to save flow visualization to {filename}.html: {e!s}"
                ) from e

        except (ValueError, RuntimeError, IOError) as e:
            raise e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error during flow visualization: {e!s}"
            ) from e
        finally:
            self._cleanup_pyvis_lib()

    def _generate_final_html(self, network_html):
        """
        Generate the final HTML content with network visualization and legend.

        Parameters
        ----------
        network_html : str
            HTML content generated by pyvis Network.

        Returns
        -------
        str
            Complete HTML content with styling and legend.

        Raises
        ------
        IOError
            If template or logo files cannot be accessed.
        ValueError
            If network_html is invalid.
        """
        if not network_html:
            raise ValueError("Invalid network HTML content")

        try:
            # Extract just the body content from the generated HTML
            current_dir = os.path.dirname(__file__)
            template_path = safe_path_join(
                "assets", "crewai_flow_visual_template.html", root=current_dir
            )
            logo_path = safe_path_join("assets", "crewai_logo.svg", root=current_dir)

            if not os.path.exists(template_path):
                raise IOError(f"Template file not found: {template_path}")
            if not os.path.exists(logo_path):
                raise IOError(f"Logo file not found: {logo_path}")

            html_handler = HTMLTemplateHandler(template_path, logo_path)
            network_body = html_handler.extract_body_content(network_html)

            # Generate the legend items HTML
            legend_items = get_legend_items(self.colors)
            legend_items_html = generate_legend_items_html(legend_items)
            return html_handler.generate_final_html(network_body, legend_items_html)
        except Exception as e:
            raise IOError(f"Failed to generate visualization HTML: {e!s}") from e

    def _cleanup_pyvis_lib(self):
        """
        Clean up the generated lib folder from pyvis.

        This method safely removes the temporary lib directory created by pyvis
        during network visualization generation.
        """
        try:
            lib_folder = safe_path_join("lib", root=os.getcwd())
            if os.path.exists(lib_folder) and os.path.isdir(lib_folder):
                import shutil

                shutil.rmtree(lib_folder)
        except ValueError as e:
            print(f"Error validating lib folder path: {e}")
        except Exception as e:
            print(f"Error cleaning up lib folder: {e}")


def plot_flow(flow, filename="flow_plot"):
    """
    Convenience function to create and save a flow visualization.

    Parameters
    ----------
    flow : Flow
        Flow instance to visualize.
    filename : str, optional
        Output filename without extension, by default "flow_plot".

    Raises
    ------
    ValueError
        If flow object or filename is invalid.
    IOError
        If file operations fail.
    """
    visualizer = FlowPlot(flow)
    visualizer.plot(filename)
