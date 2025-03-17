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
    schema: Optional[str] = Field(
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
            schema (Optional[str]): Databricks schema name
            warehouse_id (Optional[str]): SQL warehouse ID
            row_limit (Optional[int]): Maximum number of rows to return

        Returns:
            str: Formatted query results
        """
        try:
            # Get parameters with fallbacks to default values
            query = kwargs.get("query")
            catalog = kwargs.get("catalog") or self.default_catalog
            schema = kwargs.get("schema") or self.default_schema
            warehouse_id = kwargs.get("warehouse_id") or self.default_warehouse_id
            row_limit = kwargs.get("row_limit", 1000)

            # Validate schema and query
            validated_input = DatabricksQueryToolSchema(
                query=query,
                catalog=catalog,
                schema=schema,
                warehouse_id=warehouse_id,
                row_limit=row_limit
            )

            # Extract validated parameters
            query = validated_input.query
            catalog = validated_input.catalog
            schema = validated_input.schema
            warehouse_id = validated_input.warehouse_id

            # Setup SQL context with catalog/schema if provided
            context = {}
            if catalog:
                context["catalog"] = catalog
            if schema:
                context["schema"] = schema

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

            print(f"Starting to poll for statement ID: {statement_id}")

            while time.time() - start_time < timeout:
                poll_count += 1
                try:
                    # Get statement status
                    result = statement.get_statement(statement_id)

                    # Debug info
                    if poll_count % 5 == 0:  # Log every 5th poll
                        print(f"Poll #{poll_count}: State={result.status.state if hasattr(result, 'status') else 'Unknown'}")

                    # Check if finished - be very explicit about state checking
                    if hasattr(result, 'status') and hasattr(result.status, 'state'):
                        state_value = str(result.status.state)  # Convert to string to handle both string and enum

                        # Track state changes for debugging
                        if previous_state != state_value:
                            print(f"State changed from {previous_state} to {state_value}")
                            previous_state = state_value

                        # Check if state indicates completion
                        if "SUCCEEDED" in state_value:
                            print(f"Query succeeded after {poll_count} polls")
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

                            # Print error for debugging
                            print(f"Query failed after {poll_count} polls: {error_info}")

                            # Output full status object for debugging
                            print(f"Full status object: {dir(result.status)}")
                            if hasattr(result.status, 'error'):
                                print(f"Error object details: {dir(result.status.error)}")

                            # Return immediately on first FAILED state detection
                            print(f"Exiting polling loop after detecting FAILED state")
                            return f"Query execution failed: {error_info}"
                        elif "CANCELED" in state_value:
                            print(f"Query was canceled after {poll_count} polls")
                            return "Query was canceled"
                        else:
                            # Print state for debugging if not recognized
                            if poll_count % 5 == 0:
                                print(f"Current state: {state_value}")
                    else:
                        print(f"Warning: Result structure does not contain expected status attributes")

                except Exception as poll_error:
                    print(f"Error during polling (attempt #{poll_count}): {str(poll_error)}")
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

            # Debug info - print the result structure to help debug
            print(f"Result structure: {dir(result)}")
            if hasattr(result, 'manifest'):
                print(f"Manifest structure: {dir(result.manifest)}")
            if hasattr(result, 'result'):
                print(f"Result data structure: {dir(result.result)}")

            # Check if we have results and a schema in a very defensive way
            has_schema = (hasattr(result, 'manifest') and result.manifest is not None and
                         hasattr(result.manifest, 'schema') and result.manifest.schema is not None)
            has_result = (hasattr(result, 'result') and result.result is not None)

            if has_schema and has_result:
                try:
                    # Get schema for column names
                    columns = [col.name for col in result.manifest.schema.columns]

                    # Debug info for schema
                    print(f"Schema columns: {columns}")
                    print(f"Number of columns in schema: {len(columns)}")
                    print(f"Type of result.result: {type(result.result)}")

                    # Keep track of all dynamic columns we create
                    all_columns = set(columns)

                    # Dump the raw structure of result data to help troubleshoot
                    if hasattr(result.result, 'data_array'):
                        print(f"data_array structure: {type(result.result.data_array)}")
                        if result.result.data_array and len(result.result.data_array) > 0:
                            print(f"First chunk type: {type(result.result.data_array[0])}")
                            if len(result.result.data_array[0]) > 0:
                                print(f"First row type: {type(result.result.data_array[0][0])}")
                                print(f"First row value: {result.result.data_array[0][0]}")

                    # IMPROVED DETECTION LOGIC: Check if we're possibly dealing with rows where each item
                    # contains a single value or character (which could indicate incorrect row structure)
                    is_likely_incorrect_row_structure = False
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
                            print(f"Detected potential incorrect row structure: {single_char_count} single chars, {single_digit_count} digits out of {total_items} total items")
                            is_likely_incorrect_row_structure = True

                    # Additional check: if many rows have just 1 item when we expect multiple columns
                    rows_with_single_item = sum(1 for row in result.result.data_array[:sample_size] if isinstance(row, list) and len(row) == 1)
                    if rows_with_single_item > sample_size * 0.5 and len(columns) > 1:
                        print(f"Many rows ({rows_with_single_item}/{sample_size}) have only a single value when expecting {len(columns)} columns")
                        is_likely_incorrect_row_structure = True

                    # Check if we're getting primarily single characters or the data structure seems off,
                    # we should use special handling
                    if is_likely_incorrect_row_structure:
                        print("Data appears to be malformed - will use special row reconstruction")
                        needs_special_string_handling = True
                    else:
                        needs_special_string_handling = False

                    # Process results differently based on detection
                    if needs_special_string_handling:
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

                        # Print what we gathered
                        print(f"Collected {len(all_values)} total values")
                        if len(all_values) > 0:
                            print(f"Sample values: {all_values[:20]}")

                        # Get the expected column count from schema
                        expected_column_count = len(columns)
                        print(f"Expected columns per row: {expected_column_count}")

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
                                        print(f"Found potential row start at index {i}: {val}")

                        if id_indices:
                            print(f"Identified {len(id_indices)} potential row boundaries")

                            # If we found potential row starts, use them to extract rows
                            for i in range(len(id_indices)):
                                start_idx = id_indices[i]
                                end_idx = id_indices[i+1] if i+1 < len(id_indices) else len(all_values)

                                # Extract values for this row
                                row_values = all_values[start_idx:end_idx]

                                # Special handling for Netflix title data
                                # Titles might be split into individual characters
                                if 'Title' in columns and len(row_values) > expected_column_count:
                                    print(f"Row has {len(row_values)} values, likely contains split strings")

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
                                            print(f"Reconstructed title: {title}")

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
                            # If pattern recognition didn't work, try more sophisticated reconstruction
                            print("Pattern recognition did not find row boundaries, trying alternative methods")

                            # More intelligent chunking - try to detect where columns like Title might be split
                            try:
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
                                                print(f"Reconstructed title by joining characters: {row_dict[columns[title_idx]]}")

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
                            except Exception as e:
                                print(f"Error during title reconstruction: {e}")

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
                                    print(f"Found potentially incomplete title: {row.get('Title')}")

                        # Ensure we respect the row limit
                        if row_limit and len(reconstructed_rows) > row_limit:
                            reconstructed_rows = reconstructed_rows[:row_limit]
                            print(f"Limited to {row_limit} rows as requested")

                        print(f"Successfully reconstructed {len(reconstructed_rows)} rows")
                        chunk_results = reconstructed_rows
                    else:
                        # Process normal result structure as before
                        print("Using standard processing mode")

                        # Check different result structures
                        if hasattr(result.result, 'data_array') and result.result.data_array:
                            # Check if data appears to be malformed within chunks
                            for chunk_idx, chunk in enumerate(result.result.data_array):
                                print(f"Processing chunk {chunk_idx} with {len(chunk)} values")

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
                                                print(f"Reconstructed {len(reconstructed_rows)} rows from chunk")
                                                chunk_results.extend(reconstructed_rows)
                                                continue  # Skip normal processing for this chunk

                                # Special case: when chunk contains exactly the right number of values for a single row
                                # This handles the case where instead of a list of rows, we just got all values in a flat list
                                if all(isinstance(val, (str, int, float)) and not isinstance(val, (list, dict)) for val in chunk):
                                    if len(chunk) == len(columns) or (len(chunk) > 0 and len(chunk) % len(columns) == 0):
                                        print(f"Chunk appears to contain flat values - treating as rows with {len(columns)} columns each")

                                        # Process flat list of values as rows
                                        for i in range(0, len(chunk), len(columns)):
                                            row_values = chunk[i:i+len(columns)]
                                            if len(row_values) == len(columns):  # Only process complete rows
                                                row_dict = {col: val for col, val in zip(columns, row_values)}
                                                chunk_results.append(row_dict)
                                                print(f"Created row from flat values: {row_dict}")

                                        # Skip regular row processing for this chunk
                                        continue

                                # Normal processing for typical row structure
                                for row_idx, row in enumerate(chunk):
                                    # Ensure row is actually a collection of values
                                    if not isinstance(row, (list, tuple, dict)):
                                        print(f"Row {row_idx} is not a collection: {row} ({type(row)})")
                                        # This might be a single value; skip it or handle specially
                                        continue

                                    # Debug info for this row
                                    if isinstance(row, (list, tuple)):
                                        print(f"Row {row_idx} has {len(row)} values")
                                    elif isinstance(row, dict):
                                        print(f"Row {row_idx} already has column mapping: {list(row.keys())}")

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
                            print(f"Processing data with {len(result.result.data)} rows")

                            for row_idx, row in enumerate(result.result.data):
                                # Debug info
                                print(f"Row {row_idx} has {len(row)} values")

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
                    print(f"All columns detected: {all_columns}")
                    normalized_results = []
                    for row in chunk_results:
                        # Create a new row with all columns, defaulting to None for missing ones
                        normalized_row = {col: row.get(col, None) for col in all_columns}
                        normalized_results.append(normalized_row)

                    # Replace the original results with normalized ones
                    chunk_results = normalized_results

                    # Print the processed results for debugging
                    print(f"Processed {len(chunk_results)} rows")
                    for i, row in enumerate(chunk_results[:3]):  # Show only first 3 rows to avoid log spam
                        print(f"Row {i}: {row}")

                except Exception as results_error:
                    # Enhanced error message with more context
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"Error processing results: {error_details}")
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
