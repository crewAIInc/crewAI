import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

class DatabricksQueryToolSchema(BaseModel):
    """Input schema for DatabricksQueryTool."""

    query: str = Field(
        ..., description="SQL query to execute against the Databricks workspace table"
    )
    catalog: Optional[str] = Field(
        None, description="Databricks catalog name (optional, defaults to configured catalog)"
    )
    db_schema: Optional[str] = Field(
        None, description="Databricks schema name (optional, defaults to configured schema)"
    )
    warehouse_id: Optional[str] = Field(
        None, description="Databricks SQL warehouse ID (optional, defaults to configured warehouse)"
    )
    row_limit: Optional[int] = Field(
        1000, description="Maximum number of rows to return (default: 1000)"
    )

    @model_validator(mode='after')
    def validate_input(self) -> 'DatabricksQueryToolSchema':
        """Validate the input parameters."""
        # Ensure the query is not empty
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")

        # Add a LIMIT clause to the query if row_limit is provided and query doesn't have one
        if self.row_limit and "limit" not in self.query.lower():
            self.query = f"{self.query.rstrip(';')} LIMIT {self.row_limit};"

        return self


class DatabricksQueryTool(BaseTool):
    """
    A tool for querying Databricks workspace tables using SQL.

    This tool executes SQL queries against Databricks tables and returns the results.
    It requires Databricks authentication credentials to be set as environment variables.

    Authentication can be provided via:
    - Databricks CLI profile: Set DATABRICKS_CONFIG_PROFILE environment variable
    - Direct credentials: Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables

    Example:
        >>> tool = DatabricksQueryTool()
        >>> results = tool.run(query="SELECT * FROM my_table LIMIT 10")
    """

    name: str = "Databricks SQL Query"
    description: str = (
        "Execute SQL queries against Databricks workspace tables and return the results."
        " Provide a 'query' parameter with the SQL query to execute."
    )
    args_schema: Type[BaseModel] = DatabricksQueryToolSchema

    # Optional default parameters
    default_catalog: Optional[str] = None
    default_schema: Optional[str] = None
    default_warehouse_id: Optional[str] = None

    _workspace_client: Optional["WorkspaceClient"] = None

    def __init__(
        self,
        default_catalog: Optional[str] = None,
        default_schema: Optional[str] = None,
        default_warehouse_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DatabricksQueryTool.

        Args:
            default_catalog (Optional[str]): Default catalog to use for queries.
            default_schema (Optional[str]): Default schema to use for queries.
            default_warehouse_id (Optional[str]): Default SQL warehouse ID to use.
            **kwargs: Additional keyword arguments passed to BaseTool.
        """
        super().__init__(**kwargs)
        self.default_catalog = default_catalog
        self.default_schema = default_schema
        self.default_warehouse_id = default_warehouse_id
        self._validate_credentials()

    def _validate_credentials(self) -> None:
        """Validate that Databricks credentials are available."""
        has_profile = "DATABRICKS_CONFIG_PROFILE" in os.environ
        has_direct_auth = "DATABRICKS_HOST" in os.environ and "DATABRICKS_TOKEN" in os.environ

        if not (has_profile or has_direct_auth):
            raise ValueError(
                "Databricks authentication credentials are required. "
                "Set either DATABRICKS_CONFIG_PROFILE or both DATABRICKS_HOST and DATABRICKS_TOKEN environment variables."
            )

    @property
    def workspace_client(self) -> "WorkspaceClient":
        """Get or create a Databricks WorkspaceClient instance."""
        if self._workspace_client is None:
            try:
                from databricks.sdk import WorkspaceClient
                self._workspace_client = WorkspaceClient()
            except ImportError:
                raise ImportError(
                    "`databricks-sdk` package not found, please run `uv add databricks-sdk`"
                )
        return self._workspace_client

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format query results as a readable string."""
        if not results:
            return "Query returned no results."

        # Get column names from the first row
        if not results[0]:
            return "Query returned empty rows with no columns."

        columns = list(results[0].keys())

        # If we have rows but they're all empty, handle that case
        if not columns:
            return "Query returned rows but with no column data."

        # Calculate column widths based on data
        col_widths = {col: len(col) for col in columns}
        for row in results:
            for col in columns:
                # Convert value to string and get its length
                # Handle None values gracefully
                value_str = str(row[col]) if row[col] is not None else "NULL"
                col_widths[col] = max(col_widths[col], len(value_str))

        # Create header row
        header = " | ".join(f"{col:{col_widths[col]}}" for col in columns)
        separator = "-+-".join("-" * col_widths[col] for col in columns)

        # Format data rows
        data_rows = []
        for row in results:
            # Handle None values by displaying "NULL"
            row_values = {col: str(row[col]) if row[col] is not None else "NULL" for col in columns}
            data_row = " | ".join(f"{row_values[col]:{col_widths[col]}}" for col in columns)
            data_rows.append(data_row)

        # Add row count information
        result_info = f"({len(results)} row{'s' if len(results) != 1 else ''} returned)"

        # Combine all parts
        return f"{header}\n{separator}\n" + "\n".join(data_rows) + f"\n\n{result_info}"

    def _run(
        self,
        **kwargs: Any,
    ) -> str:
        """
        Execute a SQL query against Databricks and return the results.

        Args:
            query (str): SQL query to execute
            catalog (Optional[str]): Databricks catalog name
            db_schema (Optional[str]): Databricks schema name
            warehouse_id (Optional[str]): SQL warehouse ID
            row_limit (Optional[int]): Maximum number of rows to return

        Returns:
            str: Formatted query results
        """
        try:
            # Get parameters with fallbacks to default values
            query = kwargs.get("query")
            catalog = kwargs.get("catalog") or self.default_catalog
            db_schema = kwargs.get("db_schema") or self.default_schema
            warehouse_id = kwargs.get("warehouse_id") or self.default_warehouse_id
            row_limit = kwargs.get("row_limit", 1000)

            # Validate schema and query
            validated_input = DatabricksQueryToolSchema(
                query=query,
                catalog=catalog,
                db_schema=db_schema,
                warehouse_id=warehouse_id,
                row_limit=row_limit
            )

            # Extract validated parameters
            query = validated_input.query
            catalog = validated_input.catalog
            db_schema = validated_input.db_schema
            warehouse_id = validated_input.warehouse_id

            # Setup SQL context with catalog/schema if provided
            context = {}
            if catalog:
                context["catalog"] = catalog
            if db_schema:
                context["schema"] = db_schema

            # Execute query
            statement = self.workspace_client.statement_execution

            try:
                # Execute the statement
                execution = statement.execute_statement(
                    warehouse_id=warehouse_id,
                    statement=query,
                    **context
                )

                statement_id = execution.statement_id
            except Exception as execute_error:
                # Handle immediate execution errors
                return f"Error starting query execution: {str(execute_error)}"

            # Poll for results with better error handling
            import time
            result = None
            timeout = 300  # 5 minutes timeout
            start_time = time.time()
            poll_count = 0
            previous_state = None  # Track previous state to detect changes

            while time.time() - start_time < timeout:
                poll_count += 1
                try:
                    # Get statement status
                    result = statement.get_statement(statement_id)

                    # Check if finished - be very explicit about state checking
                    if hasattr(result, 'status') and hasattr(result.status, 'state'):
                        state_value = str(result.status.state)  # Convert to string to handle both string and enum

                        # Track state changes for debugging
                        if previous_state != state_value:
                            previous_state = state_value

                        # Check if state indicates completion
                        if "SUCCEEDED" in state_value:
                            break
                        elif "FAILED" in state_value:
                            # Extract error message with more robust handling
                            error_info = "No detailed error info"
                            try:
                                # First try direct access to error.message
                                if hasattr(result.status, 'error') and result.status.error:
                                    if hasattr(result.status.error, 'message'):
                                        error_info = result.status.error.message
                                    # Some APIs may have a different structure
                                    elif hasattr(result.status.error, 'error_message'):
                                        error_info = result.status.error.error_message
                                    # Last resort, try to convert the whole error object to string
                                    else:
                                        error_info = str(result.status.error)
                            except Exception as err_extract_error:
                                # If all else fails, try to get any info we can
                                error_info = f"Error details unavailable: {str(err_extract_error)}"

                            # Return immediately on first FAILED state detection
                            return f"Query execution failed: {error_info}"
                        elif "CANCELED" in state_value:
                            return "Query was canceled"

                except Exception as poll_error:
                    # Don't immediately fail - try again a few times
                    if poll_count > 3:
                        return f"Error checking query status: {str(poll_error)}"

                # Wait before polling again
                time.sleep(2)

            # Check if we timed out
            if result is None:
                return "Query returned no result (likely timed out or failed)"

            if not hasattr(result, 'status') or not hasattr(result.status, 'state'):
                return "Query completed but returned an invalid result structure"

            # Convert state to string for comparison
            state_value = str(result.status.state)
            if not any(state in state_value for state in ["SUCCEEDED", "FAILED", "CANCELED"]):
                return f"Query timed out after 5 minutes (last state: {state_value})"

            # Get results - adapt this based on the actual structure of the result object
            chunk_results = []

            # Check if we have results and a schema in a very defensive way
            has_schema = (hasattr(result, 'manifest') and result.manifest is not None and
                         hasattr(result.manifest, 'schema') and result.manifest.schema is not None)
            has_result = (hasattr(result, 'result') and result.result is not None)

            if has_schema and has_result:
                try:
                    # Get schema for column names
                    columns = [col.name for col in result.manifest.schema.columns]

                    # Debug info for schema

                    # Keep track of all dynamic columns we create
                    all_columns = set(columns)

                    # Dump the raw structure of result data to help troubleshoot
                    if hasattr(result.result, 'data_array'):
                        # Add defensive check for None data_array
                        if result.result.data_array is None:
                            print("data_array is None - likely an empty result set or DDL query")
                            # Return empty result handling rather than trying to process null data
                            return "Query executed successfully (no data returned)"

                    # IMPROVED DETECTION LOGIC: Check if we're possibly dealing with rows where each item
                    # contains a single value or character (which could indicate incorrect row structure)
                    is_likely_incorrect_row_structure = False

                    # Only try to analyze sample if data_array exists and has content
                    if hasattr(result.result, 'data_array') and result.result.data_array and len(result.result.data_array) > 0 and len(result.result.data_array[0]) > 0:
                        sample_size = min(20, len(result.result.data_array[0]))

                        if sample_size > 0:
                            single_char_count = 0
                            single_digit_count = 0
                            total_items = 0

                            for i in range(sample_size):
                                val = result.result.data_array[0][i]
                                total_items += 1
                                if isinstance(val, str) and len(val) == 1 and not val.isdigit():
                                    single_char_count += 1
                                elif isinstance(val, str) and len(val) == 1 and val.isdigit():
                                    single_digit_count += 1

                            # If a significant portion of the first values are single characters or digits,
                            # this likely indicates data is being incorrectly structured
                            if total_items > 0 and (single_char_count + single_digit_count) / total_items > 0.5:
                                is_likely_incorrect_row_structure = True

                    # Additional check: if many rows have just 1 item when we expect multiple columns
                    rows_with_single_item = 0
                    if hasattr(result.result, 'data_array') and result.result.data_array and len(result.result.data_array) > 0:
                        sample_size_for_rows = min(sample_size, len(result.result.data_array[0])) if 'sample_size' in locals() else min(20, len(result.result.data_array[0]))
                        rows_with_single_item = sum(1 for row in result.result.data_array[0][:sample_size_for_rows] if isinstance(row, list) and len(row) == 1)
                        if rows_with_single_item > sample_size_for_rows * 0.5 and len(columns) > 1:
                            is_likely_incorrect_row_structure = True

                    # Check if we're getting primarily single characters or the data structure seems off,
                    # we should use special handling
                    if 'is_likely_incorrect_row_structure' in locals() and is_likely_incorrect_row_structure:
                        print("Data appears to be malformed - will use special row reconstruction")
                        needs_special_string_handling = True
                    else:
                        needs_special_string_handling = False

                    # Process results differently based on detection
                    if 'needs_special_string_handling' in locals() and needs_special_string_handling:
                        # We're dealing with data where the rows may be incorrectly structured
                        print("Using row reconstruction processing mode")

                        # Collect all values into a flat list
                        all_values = []
                        if hasattr(result.result, 'data_array') and result.result.data_array:
                            # Flatten all values into a single list
                            for chunk in result.result.data_array:
                                for item in chunk:
                                    if isinstance(item, (list, tuple)):
                                        all_values.extend(item)
                                    else:
                                        all_values.append(item)

                        # Get the expected column count from schema
                        expected_column_count = len(columns)

                        # Try to reconstruct rows using pattern recognition
                        reconstructed_rows = []

                        # PATTERN RECOGNITION APPROACH
                        # Look for likely indicators of row boundaries in the data
                        # For Netflix data, we expect IDs as numbers, titles as text strings, etc.

                        # Use regex pattern to identify ID columns that likely start a new row
                        import re
                        id_pattern = re.compile(r'^\d{5,9}$')  # Netflix IDs are often 5-9 digits
                        id_indices = []

                        for i, val in enumerate(all_values):
                            if isinstance(val, str) and id_pattern.match(val):
                                # This value looks like an ID, might be the start of a row
                                if i < len(all_values) - 1:
                                    next_few_values = all_values[i+1:i+5]
                                    # If following values look like they could be part of a title
                                    if any(isinstance(v, str) and len(v) > 1 for v in next_few_values):
                                        id_indices.append(i)

                        if id_indices:

                            # If we found potential row starts, use them to extract rows
                            for i in range(len(id_indices)):
                                start_idx = id_indices[i]
                                end_idx = id_indices[i+1] if i+1 < len(id_indices) else len(all_values)

                                # Extract values for this row
                                row_values = all_values[start_idx:end_idx]

                                # Special handling for Netflix title data
                                # Titles might be split into individual characters
                                if 'Title' in columns and len(row_values) > expected_column_count:

                                    # Try to reconstruct by looking for patterns
                                    # We know ID is first, then Title (which may be split)
                                    # Then other fields like Genre, etc.

                                    # Take first value as ID
                                    row_dict = {columns[0]: row_values[0]}

                                    # Look for Genre or other non-title fields to determine where title ends
                                    title_end_idx = 1
                                    for j in range(2, min(100, len(row_values))):
                                        val = row_values[j]
                                        # Check for common genres or non-title markers
                                        if isinstance(val, str) and val in ['Comedy', 'Drama', 'Action', 'Horror', 'Thriller', 'Documentary']:
                                            # Likely found the Genre field
                                            title_end_idx = j
                                            break

                                    # Reconstruct title from individual characters
                                    if title_end_idx > 1:
                                        title_chars = row_values[1:title_end_idx]
                                        # Check if they're individual characters
                                        if all(isinstance(c, str) and len(c) == 1 for c in title_chars):
                                            title = ''.join(title_chars)
                                            row_dict['Title'] = title

                                            # Assign remaining values to columns
                                            remaining_values = row_values[title_end_idx:]
                                            for j, col_name in enumerate(columns[2:], 2):
                                                if j-2 < len(remaining_values):
                                                    row_dict[col_name] = remaining_values[j-2]
                                                else:
                                                    row_dict[col_name] = None
                                    else:
                                        # Fallback: simple mapping
                                        for j, col_name in enumerate(columns):
                                            if j < len(row_values):
                                                row_dict[col_name] = row_values[j]
                                            else:
                                                row_dict[col_name] = None
                                else:
                                    # Standard mapping
                                    row_dict = {}
                                    for j, col_name in enumerate(columns):
                                        if j < len(row_values):
                                            row_dict[col_name] = row_values[j]
                                        else:
                                            row_dict[col_name] = None

                                reconstructed_rows.append(row_dict)
                        else:
                            # More intelligent chunking - try to detect where columns like Title might be split
                            title_idx = columns.index('Title') if 'Title' in columns else -1

                            if title_idx >= 0:
                                print("Attempting title reconstruction method")
                                # Try to detect if title is split across multiple values
                                i = 0
                                while i < len(all_values):
                                    # Check if this could be an ID (start of a row)
                                    if isinstance(all_values[i], str) and id_pattern.match(all_values[i]):
                                        row_dict = {columns[0]: all_values[i]}
                                        i += 1

                                        # Try to reconstruct title if it appears to be split
                                        title_chars = []
                                        while (i < len(all_values) and
                                                isinstance(all_values[i], str) and
                                                len(all_values[i]) <= 1 and
                                                len(title_chars) < 100):  # Cap title length
                                            title_chars.append(all_values[i])
                                            i += 1

                                        if title_chars:
                                            row_dict[columns[title_idx]] = ''.join(title_chars)

                                        # Add remaining fields
                                        for j in range(title_idx + 1, len(columns)):
                                            if i < len(all_values):
                                                row_dict[columns[j]] = all_values[i]
                                                i += 1
                                            else:
                                                row_dict[columns[j]] = None

                                        reconstructed_rows.append(row_dict)
                                    else:
                                        i += 1

                        # If we still don't have rows, use simple chunking as fallback
                        if not reconstructed_rows:
                            print("Falling back to basic chunking approach")
                            chunks = [all_values[i:i+expected_column_count] for i in range(0, len(all_values), expected_column_count)]

                            for chunk in chunks:
                                # Skip chunks that seem to be partial/incomplete rows
                                if len(chunk) < expected_column_count * 0.75:  # Allow for some missing values
                                    continue

                                row_dict = {}

                                # Map values to column names
                                for i, col in enumerate(columns):
                                    if i < len(chunk):
                                        row_dict[col] = chunk[i]
                                    else:
                                        row_dict[col] = None

                                reconstructed_rows.append(row_dict)

                        # Apply post-processing to fix known issues
                        if reconstructed_rows and 'Title' in columns:
                            print("Applying post-processing to improve data quality")
                            for row in reconstructed_rows:
                                # Fix titles that might still have issues
                                if isinstance(row.get('Title'), str) and len(row.get('Title')) <= 1:
                                    # This is likely still a fragmented title - mark as potentially incomplete
                                    row['Title'] = f"[INCOMPLETE] {row.get('Title')}"

                        # Ensure we respect the row limit
                        if row_limit and len(reconstructed_rows) > row_limit:
                            reconstructed_rows = reconstructed_rows[:row_limit]

                        chunk_results = reconstructed_rows
                    else:
                        # Process normal result structure as before
                        print("Using standard processing mode")

                        # Check different result structures
                        if hasattr(result.result, 'data_array') and result.result.data_array:
                            # Check if data appears to be malformed within chunks
                            for chunk_idx, chunk in enumerate(result.result.data_array):

                                # Check if chunk might actually contain individual columns of a single row
                                # This is another way data might be malformed - check the first few values
                                if len(chunk) > 0 and len(columns) > 1:
                                    # If there seems to be a mismatch between chunk structure and expected columns
                                    first_few_values = chunk[:min(5, len(chunk))]
                                    if all(isinstance(val, (str, int, float)) and not isinstance(val, (list, dict)) for val in first_few_values):
                                        if len(chunk) > len(columns) * 3:  # Heuristic: if chunk has way more items than columns
                                            print("Chunk appears to contain individual values rather than rows - switching to row reconstruction")

                                            # This chunk might actually be values of multiple rows - try to reconstruct
                                            values = chunk  # All values in this chunk
                                            reconstructed_rows = []

                                            # Try to create rows based on expected column count
                                            for i in range(0, len(values), len(columns)):
                                                if i + len(columns) <= len(values):  # Ensure we have enough values
                                                    row_values = values[i:i+len(columns)]
                                                    row_dict = {col: val for col, val in zip(columns, row_values)}
                                                    reconstructed_rows.append(row_dict)

                                            if reconstructed_rows:
                                                chunk_results.extend(reconstructed_rows)
                                                continue  # Skip normal processing for this chunk

                                # Special case: when chunk contains exactly the right number of values for a single row
                                # This handles the case where instead of a list of rows, we just got all values in a flat list
                                if all(isinstance(val, (str, int, float)) and not isinstance(val, (list, dict)) for val in chunk):
                                    if len(chunk) == len(columns) or (len(chunk) > 0 and len(chunk) % len(columns) == 0):

                                        # Process flat list of values as rows
                                        for i in range(0, len(chunk), len(columns)):
                                            row_values = chunk[i:i+len(columns)]
                                            if len(row_values) == len(columns):  # Only process complete rows
                                                row_dict = {col: val for col, val in zip(columns, row_values)}
                                                chunk_results.append(row_dict)

                                        # Skip regular row processing for this chunk
                                        continue

                                # Normal processing for typical row structure
                                for row_idx, row in enumerate(chunk):
                                    # Ensure row is actually a collection of values
                                    if not isinstance(row, (list, tuple, dict)):
                                        # This might be a single value; skip it or handle specially
                                        continue

                                    # Convert each row to a dictionary with column names as keys
                                    row_dict = {}

                                    # Handle dict rows directly
                                    if isinstance(row, dict):
                                        # Use the existing column mapping
                                        row_dict = dict(row)
                                    elif isinstance(row, (list, tuple)):
                                        # Map list of values to columns
                                        for i, val in enumerate(row):
                                            if i < len(columns):  # Only process if we have a matching column
                                                row_dict[columns[i]] = val
                                            else:
                                                # Extra values without column names
                                                dynamic_col = f"Column_{i}"
                                                row_dict[dynamic_col] = val
                                                all_columns.add(dynamic_col)

                                    # If we have fewer values than columns, set missing values to None
                                    for col in columns:
                                        if col not in row_dict:
                                            row_dict[col] = None

                                    chunk_results.append(row_dict)

                        elif hasattr(result.result, 'data') and result.result.data:
                            # Alternative data structure

                            for row_idx, row in enumerate(result.result.data):
                                # Debug info

                                # Safely create dictionary matching column names to values
                                row_dict = {}
                                for i, val in enumerate(row):
                                    if i < len(columns):  # Only process if we have a matching column
                                        row_dict[columns[i]] = val
                                    else:
                                        # Extra values without column names
                                        dynamic_col = f"Column_{i}"
                                        row_dict[dynamic_col] = val
                                        all_columns.add(dynamic_col)

                                # If we have fewer values than columns, set missing values to None
                                for i, col in enumerate(columns):
                                    if i >= len(row):
                                        row_dict[col] = None

                                chunk_results.append(row_dict)

                    # After processing all rows, ensure all rows have all columns
                    normalized_results = []
                    for row in chunk_results:
                        # Create a new row with all columns, defaulting to None for missing ones
                        normalized_row = {col: row.get(col, None) for col in all_columns}
                        normalized_results.append(normalized_row)

                    # Replace the original results with normalized ones
                    chunk_results = normalized_results

                except Exception as results_error:
                    # Enhanced error message with more context
                    import traceback
                    error_details = traceback.format_exc()
                    return f"Error processing query results: {str(results_error)}\n\nDetails:\n{error_details}"

            # If we have no results but the query succeeded (e.g., for DDL statements)
            if not chunk_results and hasattr(result, 'status'):
                state_value = str(result.status.state)
                if "SUCCEEDED" in state_value:
                    return "Query executed successfully (no results to display)"

            # Format and return results
            return self._format_results(chunk_results)

        except Exception as e:
            # Include more details in the error message to help with debugging
            import traceback
            error_details = traceback.format_exc()
            return f"Error executing Databricks query: {str(e)}\n\nDetails:\n{error_details}"
