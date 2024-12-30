import asyncio
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
    cast,
)

from blinker import Signal
from pydantic import BaseModel, ValidationError

from crewai.flow.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.flow.core_flow_utils import get_possible_return_constants
from crewai.telemetry import Telemetry

T = TypeVar("T", bound=Union[BaseModel, Dict[str, Any]])


def start(condition: Optional[Union[str, dict, Callable]] = None) -> Callable:
    """Marks a method as a flow starting point, optionally triggered by other methods.
    
    Args:
        condition: The condition that triggers this method. Can be:
            - str: Name of the triggering method
            - dict: Dictionary with 'type' and 'methods' keys for complex conditions
            - Callable: A function reference
            - None: No trigger condition (default)
    
    Returns:
        Callable: The decorated function that will serve as a flow starting point.
        
    Raises:
        ValueError: If the condition format is invalid.
        
    Example:
        >>> @start()  # No condition
        >>> def begin_flow():
        >>>     pass
        >>>
        >>> @start("method_name")  # Triggered by specific method
        >>> def conditional_start():
        >>>     pass
    """
    def decorator(func):
        func.__is_start_method__ = True
        if condition is not None:
            if isinstance(condition, str):
                func.__trigger_methods__ = [condition]
                func.__condition_type__ = "OR"
            elif (
                isinstance(condition, dict)
                and "type" in condition
                and "methods" in condition
            ):
                func.__trigger_methods__ = condition["methods"]
                func.__condition_type__ = condition["type"]
            elif callable(condition) and hasattr(condition, "__name__"):
                func.__trigger_methods__ = [condition.__name__]
                func.__condition_type__ = "OR"
            else:
                raise ValueError(
                    "Condition must be a method, string, or a result of or_() or and_()"
                )
        return func

    return decorator


def listen(condition: Union[str, dict, Callable]) -> Callable:
    """Marks a method to execute when specified conditions/methods complete.
    
    Args:
        condition: The condition that triggers this method. Can be:
            - str: Name of the triggering method
            - dict: Dictionary with 'type' and 'methods' keys for complex conditions
            - Callable: A function reference
    
    Returns:
        Callable: The decorated function that will execute when conditions are met.
        
    Raises:
        ValueError: If the condition format is invalid.
        
    Example:
        >>> @listen("start_method")  # Listen to single method
        >>> def on_start():
        >>>     pass
        >>>
        >>> @listen(and_("method1", "method2"))  # Listen with AND condition
        >>> def on_both_complete():
        >>>     pass
    """
    def decorator(func):
        if isinstance(condition, str):
            func.__trigger_methods__ = [condition]
            func.__condition_type__ = "OR"
        elif (
            isinstance(condition, dict)
            and "type" in condition
            and "methods" in condition
        ):
            func.__trigger_methods__ = condition["methods"]
            func.__condition_type__ = condition["type"]
        elif callable(condition) and hasattr(condition, "__name__"):
            func.__trigger_methods__ = [condition.__name__]
            func.__condition_type__ = "OR"
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return func

    return decorator


def router(condition: Union[str, dict, Callable]) -> Callable:
    """Marks a method as a router to direct flow based on its return value.
    
    A router method can return different string values that trigger different
    subsequent methods, allowing for dynamic flow control.
    
    Args:
        condition: The condition that triggers this router. Can be:
            - str: Name of the triggering method
            - dict: Dictionary with 'type' and 'methods' keys for complex conditions
            - Callable: A function reference
    
    Returns:
        Callable: The decorated function that will serve as a router.
        
    Raises:
        ValueError: If the condition format is invalid.
        
    Example:
        >>> @router("process_data")
        >>> def route_result(result):
        >>>     if result.success:
        >>>         return "handle_success"
        >>>     return "handle_error"
    """
    def decorator(func):
        func.__is_router__ = True
        if isinstance(condition, str):
            func.__trigger_methods__ = [condition]
            func.__condition_type__ = "OR"
        elif (
            isinstance(condition, dict)
            and "type" in condition
            and "methods" in condition
        ):
            func.__trigger_methods__ = condition["methods"]
            func.__condition_type__ = condition["type"]
        elif callable(condition) and hasattr(condition, "__name__"):
            func.__trigger_methods__ = [condition.__name__]
            func.__condition_type__ = "OR"
        else:
            raise ValueError(
                "Condition must be a method, string, or a result of or_() or and_()"
            )
        return func

    return decorator


def or_(*conditions: Union[str, dict, Callable]) -> dict:
    """Combines multiple conditions with OR logic for flow control.
    
    Args:
        *conditions: Variable number of conditions. Each can be:
            - str: Name of a method
            - dict: Dictionary with 'type' and 'methods' keys
            - Callable: A function reference
    
    Returns:
        dict: A dictionary with 'type': 'OR' and 'methods' list.
        
    Raises:
        ValueError: If any condition is invalid.
        
    Example:
        >>> @listen(or_("method1", "method2"))
        >>> def on_either():
        >>>     # Executes when either method1 OR method2 completes
        >>>     pass
    """
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif isinstance(condition, str):
            methods.append(condition)
        elif callable(condition):
            methods.append(getattr(condition, "__name__", repr(condition)))
        else:
            raise ValueError("Invalid condition in or_()")
    return {"type": "OR", "methods": methods}


def and_(*conditions: Union[str, dict, Callable]) -> dict:
    """Combines multiple conditions with AND logic for flow control.
    
    Args: 
        *conditions: Variable number of conditions. Each can be:
            - str: Name of a method
            - dict: Dictionary with 'type' and 'methods' keys
            - Callable: A function reference
    
    Returns:
        dict: A dictionary with 'type': 'AND' and 'methods' list.
        
    Raises:
        ValueError: If any condition is invalid.
        
    Example:
        >>> @listen(and_("method1", "method2"))
        >>> def on_both():
        >>>     # Executes when BOTH method1 AND method2 complete
        >>>     pass
    """
    methods = []
    for condition in conditions:
        if isinstance(condition, dict) and "methods" in condition:
            methods.extend(condition["methods"])
        elif isinstance(condition, str):
            methods.append(condition)
        elif callable(condition):
            methods.append(getattr(condition, "__name__", repr(condition)))
        else:
            raise ValueError("Invalid condition in and_()")
    return {"type": "AND", "methods": methods}


class FlowMeta(type):
    def __new__(mcs, name, bases, dct):
        cls = super().__new__(mcs, name, bases, dct)

        start_methods = []
        listeners = {}
        router_paths = {}
        routers = set()

        for attr_name, attr_value in dct.items():
            if hasattr(attr_value, "__is_start_method__"):
                start_methods.append(attr_name)
                if hasattr(attr_value, "__trigger_methods__"):
                    methods = attr_value.__trigger_methods__
                    condition_type = getattr(attr_value, "__condition_type__", "OR")
                    listeners[attr_name] = (condition_type, methods)
            elif hasattr(attr_value, "__trigger_methods__"):
                methods = attr_value.__trigger_methods__
                condition_type = getattr(attr_value, "__condition_type__", "OR")
                listeners[attr_name] = (condition_type, methods)
                if hasattr(attr_value, "__is_router__") and attr_value.__is_router__:
                    routers.add(attr_name)
                    possible_returns = get_possible_return_constants(attr_value)
                    if possible_returns:
                        router_paths[attr_name] = possible_returns

        setattr(cls, "_start_methods", start_methods)
        setattr(cls, "_listeners", listeners)
        setattr(cls, "_routers", routers)
        setattr(cls, "_router_paths", router_paths)

        return cls


class Flow(Generic[T], metaclass=FlowMeta):
    _telemetry = Telemetry()

    _start_methods: List[str] = []
    _listeners: Dict[str, tuple[str, List[str]]] = {}
    _routers: Set[str] = set()
    _router_paths: Dict[str, List[str]] = {}
    initial_state: Union[Type[T], T, None] = None
    event_emitter = Signal("event_emitter")

    def __class_getitem__(cls: Type["Flow"], item: Type[T]) -> Type["Flow"]:
        """Create a generic version of Flow with specified state type.
        
        Args:
            cls: The Flow class
            item: The type parameter for the flow's state
            
        Returns:
            Type["Flow"]: A new Flow class with the specified state type
            
        Example:
            >>> class MyState(BaseModel):
            >>>     value: int
            >>> 
            >>> class MyFlow(Flow[MyState]):
            >>>     pass
        """
        class _FlowGeneric(cls):  # type: ignore
            _initial_state_T = item  # type: ignore

        _FlowGeneric.__name__ = f"{cls.__name__}[{item.__name__}]"
        return _FlowGeneric

    def __init__(self) -> None:
        """Initialize a new Flow instance.
        
        Sets up internal state tracking, method registration, and telemetry.
        The flow's methods are automatically discovered and registered during initialization.
        
        Attributes initialized:
            _methods: Dictionary mapping method names to their callable objects
            _state: The flow's state object of type T
            _method_execution_counts: Tracks how many times each method has executed
            _pending_and_listeners: Tracks methods waiting for AND conditions
            _method_outputs: List of all outputs from executed methods
        """
        self._methods: Dict[str, Callable] = {}
        self._state: T = self._create_initial_state()
        self._method_execution_counts: Dict[str, int] = {}
        self._pending_and_listeners: Dict[str, Set[str]] = {}
        self._method_outputs: List[Any] = []

        self._telemetry.flow_creation_span(self.__class__.__name__)

        for method_name in dir(self):
            if callable(getattr(self, method_name)) and not method_name.startswith(
                "__"
            ):
                self._methods[method_name] = getattr(self, method_name)

    def _create_initial_state(self) -> T:
        """Create the initial state for the flow.
        
        The state is created based on the following priority:
        1. If initial_state is None and _initial_state_T exists (generic type), use that
        2. If initial_state is None, return empty dict
        3. If initial_state is a type, instantiate it
        4. Otherwise, use initial_state as-is
        
        Returns:
            T: The initial state object of type T
            
        Note:
            The type T can be either a Pydantic BaseModel or a dictionary.
        """
        if self.initial_state is None and hasattr(self, "_initial_state_T"):
            return self._initial_state_T()  # type: ignore
        if self.initial_state is None:
            return {}  # type: ignore
        elif isinstance(self.initial_state, type):
            return self.initial_state()
        else:
            return self.initial_state

    @property
    def state(self) -> T:
        """Get the current state of the flow.
        
        Returns:
            T: The current state object, either a Pydantic model or dictionary
        """
        return self._state

    @property
    def method_outputs(self) -> List[Any]:
        """Get the list of all outputs from executed methods.
        
        Returns:
            List[Any]: A list containing the output values from all executed flow methods,
                      in order of execution.
        """
        return self._method_outputs

    def _initialize_state(self, inputs: Dict[str, Any]) -> None:
        if isinstance(self._state, BaseModel):
            # Structured state
            try:

                def create_model_with_extra_forbid(
                    base_model: Type[BaseModel],
                ) -> Type[BaseModel]:
                    class ModelWithExtraForbid(base_model):  # type: ignore
                        model_config = base_model.model_config.copy()
                        model_config["extra"] = "forbid"

                    return ModelWithExtraForbid

                ModelWithExtraForbid = create_model_with_extra_forbid(
                    self._state.__class__
                )
                self._state = cast(
                    T, ModelWithExtraForbid(**{**self._state.model_dump(), **inputs})
                )
            except ValidationError as e:
                raise ValueError(f"Invalid inputs for structured state: {e}") from e
        elif isinstance(self._state, dict):
            self._state.update(inputs)
        else:
            raise TypeError("State must be a BaseModel instance or a dictionary.")

    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        self.event_emitter.send(
            self,
            event=FlowStartedEvent(
                type="flow_started",
                flow_name=self.__class__.__name__,
            ),
        )

        if inputs is not None:
            self._initialize_state(inputs)
        return asyncio.run(self.kickoff_async())

    async def kickoff_async(self, inputs: Optional[Dict[str, Any]] = None) -> Any:
        if not self._start_methods:
            raise ValueError("No start method defined")

        self._telemetry.flow_execution_span(
            self.__class__.__name__, list(self._methods.keys())
        )

        tasks = [
            self._execute_start_method(start_method)
            for start_method in self._start_methods
        ]
        await asyncio.gather(*tasks)

        final_output = self._method_outputs[-1] if self._method_outputs else None

        self.event_emitter.send(
            self,
            event=FlowFinishedEvent(
                type="flow_finished",
                flow_name=self.__class__.__name__,
                result=final_output,
            ),
        )
        return final_output

    async def _execute_start_method(self, start_method_name: str) -> None:
        result = await self._execute_method(
            start_method_name, self._methods[start_method_name]
        )
        await self._execute_listeners(start_method_name, result)

    async def _execute_method(
        self, method_name: str, method: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        result = (
            await method(*args, **kwargs)
            if asyncio.iscoroutinefunction(method)
            else method(*args, **kwargs)
        )
        self._method_outputs.append(result)
        self._method_execution_counts[method_name] = (
            self._method_execution_counts.get(method_name, 0) + 1
        )
        return result

    async def _execute_listeners(self, trigger_method: str, result: Any) -> None:
        """Execute all listener methods triggered by a completed method.
        
        This method handles both router and non-router listeners in a specific order:
        1. First executes all triggered router methods sequentially until no more routers
           are triggered
        2. Then executes all regular listeners in parallel
        
        Args:
            trigger_method: The name of the method that completed execution
            result: The result value from the triggering method
            
        Note:
            Router methods are executed sequentially to ensure proper flow control,
            while regular listeners are executed concurrently for better performance.
            This provides fine-grained control over the execution flow while
            maintaining efficiency.
        """
        # First, handle routers repeatedly until no router triggers anymore
        while True:
            routers_triggered = self._find_triggered_methods(
                trigger_method, router_only=True
            )
            if not routers_triggered:
                break
            for router_name in routers_triggered:
                await self._execute_single_listener(router_name, result)
                # After executing router, the router's result is the path
                # The last router executed sets the trigger_method
                # The router result is the last element in self._method_outputs
                trigger_method = self._method_outputs[-1]

        # Now that no more routers are triggered by current trigger_method,
        # execute normal listeners
        listeners_triggered = self._find_triggered_methods(
            trigger_method, router_only=False
        )
        if listeners_triggered:
            tasks = [
                self._execute_single_listener(listener_name, result)
                for listener_name in listeners_triggered
            ]
            await asyncio.gather(*tasks)

    def _find_triggered_methods(
        self, trigger_method: str, router_only: bool
    ) -> List[str]:
        """Find all methods that should be triggered based on completed method and type.
        
        Provides precise control over method triggering by handling both OR and AND
        conditions separately for router and non-router methods.
        
        Args:
            trigger_method: The name of the method that completed execution
            router_only: If True, only find router methods; if False, only regular
                        listeners
            
        Returns:
            List[str]: Names of methods that should be executed next
            
        Note:
            This method implements sophisticated flow control by: 
            1. Filtering methods based on their router/non-router status
            2. Handling OR conditions for immediate triggering
            3. Managing AND conditions with state tracking for complex dependencies
            
            This ensures predictable and consistent execution order in complex flows.
        """
        triggered = []
        for listener_name, (condition_type, methods) in self._listeners.items():
            is_router = listener_name in self._routers

            if router_only != is_router:
                continue

            if condition_type == "OR":
                # If the trigger_method matches any in methods, run this
                if trigger_method in methods:
                    triggered.append(listener_name)
            elif condition_type == "AND":
                # Initialize pending methods for this listener if not already done
                if listener_name not in self._pending_and_listeners:
                    self._pending_and_listeners[listener_name] = set(methods)
                # Remove the trigger method from pending methods
                if trigger_method in self._pending_and_listeners[listener_name]:
                    self._pending_and_listeners[listener_name].discard(trigger_method)

                if not self._pending_and_listeners[listener_name]:
                    # All required methods have been executed
                    triggered.append(listener_name)
                    # Reset pending methods for this listener
                    self._pending_and_listeners.pop(listener_name, None)

        return triggered

    async def _execute_single_listener(self, listener_name: str, result: Any) -> None:
        """Execute a single listener method with precise parameter handling and error tracking.
        
        Provides fine-grained control over method execution through:
        1. Automatic parameter inspection to determine if the method accepts results
        2. Event emission for execution tracking
        3. Comprehensive error handling
        4. Recursive listener execution
        
        Args:
            listener_name: The name of the listener method to execute
            result: The result from the triggering method, passed to the listener
                   if its signature accepts parameters
                   
        Note:
            This method ensures precise execution control by:
            - Inspecting method signatures to handle parameters correctly
            - Emitting events for execution tracking
            - Providing comprehensive error handling
            - Supporting both parameterized and parameter-less methods
            - Maintaining execution chain through recursive listener calls
        """
        try:
            method = self._methods[listener_name]

            self.event_emitter.send(
                self,
                event=MethodExecutionStartedEvent(
                    type="method_execution_started",
                    method_name=listener_name,
                    flow_name=self.__class__.__name__,
                ),
            )

            sig = inspect.signature(method)
            params = list(sig.parameters.values())
            method_params = [p for p in params if p.name != "self"]

            if method_params:
                listener_result = await self._execute_method(
                    listener_name, method, result
                )
            else:
                listener_result = await self._execute_method(listener_name, method)

            self.event_emitter.send(
                self,
                event=MethodExecutionFinishedEvent(
                    type="method_execution_finished",
                    method_name=listener_name,
                    flow_name=self.__class__.__name__,
                ),
            )

            # Execute listeners (and possibly routers) of this listener
            await self._execute_listeners(listener_name, listener_result)

        except Exception as e:
            print(
                f"[Flow._execute_single_listener] Error in method {listener_name}: {e}"
            )
            import traceback

            traceback.print_exc()

    def plot(self, *args, **kwargs):
        """Generate an interactive visualization of the flow's execution graph.
        
        Creates a detailed HTML visualization showing the relationships between
        methods, including start points, listeners, routers, and their
        connections. Includes telemetry tracking for flow analysis.
        
        Args:
            *args: Variable length argument list passed to plot_flow
            **kwargs: Arbitrary keyword arguments passed to plot_flow
                     
        Note:
            The visualization provides:
            - Clear representation of method relationships
            - Visual distinction between different method types
            - Interactive exploration capabilities
            - Execution path tracing
            - Telemetry tracking for flow analysis
            
        Example:
            >>> flow = MyFlow()
            >>> flow.plot("my_workflow")  # Creates my_workflow.html
        """
        from crewai.flow.flow_visualizer import plot_flow
        
        self._telemetry.flow_plotting_span(
            self.__class__.__name__, list(self._methods.keys())
        )
        return plot_flow(self, *args, **kwargs)
