#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
import itertools
import logging
from typing import Any, Callable, Generator, Iterable
import multiprocessing
from concurrent import futures
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class PipelineTask:
    id: int
    payload: Any
    comp_stamp: float = 0
    comp_time: float = 0
    exec_stamp: float = 0
    exec_time: float = 0
    comp_result: Any = None
    exec_result: Any = None


class CompileExecutePipeline:
    def __init__(
        self,
        compile_func: Callable[[Any], Any],
        execute_func: Callable[[Any], Any],
        compile_jobs: int = 0,
        execute_jobs: int = 0,
    ):
        self._compile_jobs = (
            max(1, multiprocessing.cpu_count() // 2)
            if compile_jobs < 1
            else compile_jobs
        )
        self._execute_jobs = 1 if execute_jobs < 1 else execute_jobs
        self._compile_func = compile_func
        self._execute_func = execute_func

    def _process_tasks(
        self,
        tasks: Iterable[PipelineTask],
        func: Callable[[Iterable[PipelineTask]], Generator[PipelineTask, None, None]],
        batch: int,
    ) -> Generator[PipelineTask, None, None]:
        iterator = iter(tasks)
        while batched := tuple(itertools.islice(iterator, batch)):
            yield from func(batched)

    def _compile_tasks(
        self, tasks: Iterable[PipelineTask]
    ) -> Generator[PipelineTask, None, None]:
        def compile_func(tasks: Iterable[PipelineTask]):
            def compile(task: PipelineTask) -> PipelineTask:
                task.comp_stamp = time.time()
                task.comp_result = self._compile_func(task.payload)
                task.comp_time = time.time() - task.comp_stamp
                return task

            with futures.ThreadPoolExecutor(max_workers=self._compile_jobs) as executor:
                it_map = executor.map(compile, tasks)
            yield from it_map

        yield from self._process_tasks(
            tasks,
            compile_func,
            batch=self._compile_jobs,
        )

    def _execute_tasks(
        self, tasks: Iterable[PipelineTask]
    ) -> Generator[PipelineTask, None, None]:
        def execute_func(tasks: Iterable[PipelineTask]):
            def execute(task: PipelineTask) -> PipelineTask:
                task.exec_stamp = time.time()
                task.exec_result = self._execute_func(task.comp_result)
                task.exec_time = time.time() - task.exec_stamp
                return task

            with futures.ThreadPoolExecutor(max_workers=self._execute_jobs) as executor:
                it_map = executor.map(execute, tasks)
            yield from it_map

        yield from self._process_tasks(
            tasks,
            execute_func,
            batch=self._execute_jobs,
        )

    def generate(self, payloads: Iterable[Any]) -> Generator[PipelineTask, None, None]:
        compile_jobs = self._compile_jobs
        execute_jobs = self._execute_jobs
        max_jobs = max(compile_jobs, execute_jobs)
        tasks = (
            PipelineTask(id=id, payload=payload) for id, payload in enumerate(payloads)
        )

        def compile_and_execute(
            tasks: Iterable[PipelineTask],
        ) -> Generator[PipelineTask, None, None]:
            compile_results = self._compile_tasks(tasks)
            execution_results = self._execute_tasks(compile_results)
            yield from execution_results

        yield from self._process_tasks(
            tasks,
            compile_and_execute,
            batch=max_jobs,
        )

    def run(self, payloads: Iterable[Any]) -> list[PipelineTask]:
        return list(self.generate(payloads))
