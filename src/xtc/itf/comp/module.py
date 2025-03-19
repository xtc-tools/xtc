#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from abc import ABC, abstractmethod
from typing import Any

from ..exec import Evaluator, Executor


class Module(ABC):
    """An abstract representation of an executable module.

    A Module is the final output of the compilation process, representing
    compiled code that can be executed. It is produced by a Compiler after
    applying transformations specified by a Schedule to an Implementer's
    representation of a Graph.

    Modules can be exported as shared objects for direct execution and evaluation,
    or for usage in larger applications. They can be executed and evaluated using
    Executor and Evaluator classes to measure performance and validate correctness.
    """

    @property
    @abstractmethod
    def file_type(self) -> str:
        """The module type, can be target dependent.

        Available types are: "executable", "shlib"

        Returns:
            the type of the module
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The module name.

        The module name may be used to identify a module to
        execute.

        Returns:
            the name of the module
        """
        ...

    @property
    @abstractmethod
    def payload_name(self) -> str:
        """The payload name for the module.

        The name of the payload to execute for the module.
        Generally the entry point inside the module.

        Returns:
            the name of the executable payload inside the module
        """
        ...

    @property
    @abstractmethod
    def file_name(self) -> str:
        """The storage file name of the module.

        The file name extension should match the module file type.

        Returns:
            the path to the generated module file
        """
        ...

    @abstractmethod
    def export(self) -> None:
        """Exports the module to a format suitable for execution.

        This method handles the final step of making the compiled code
        available for execution, typically by writing it to a shared
        object file or similar executable format.
        """
        ...

    @abstractmethod
    def get_evaluator(self, **kwargs: Any) -> Evaluator:
        """Returns a suitable evaluator for the module.

        Args:
            kwargs: evaluator configuration

        Returns:
            The evaluator for executing the module
        """
        ...

    @abstractmethod
    def get_executor(self, **kwargs: Any) -> Executor:
        """Returns a suitable executor for the module.

        Args:
            kwargs: executor configuration

        Returns:
            The executor for executing the module
        """
        ...
