#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#
from typing_extensions import override
from typing import Any
import threading


class SearchProgress:
    def search_start(self, ntasks: int):
        pass

    def batch_start(self, batch_ntasks: int):
        pass

    def compile_batch_start(self):
        pass

    def compile_job_start(self):
        pass

    def compile_job_end(self):
        pass

    def compile_batch_end(self):
        pass

    def execute_batch_start(self):
        pass

    def execute_job_start(self):
        pass

    def execute_job_end(self):
        pass

    def execute_batch_end(self):
        pass

    def batch_end(self):
        pass

    def search_end(self):
        pass


class SearchProgressTQDM(SearchProgress):
    def __init__(
        self,
        ncomp_per_job: int = 1,
        nexec_per_job: int = 1,
        quiet: bool = False,
        prefix: str = "",
        position: int = 0,
        progress_cls: str = "tqdm",
    ):
        self.ncomp_per_job = ncomp_per_job
        self.nexec_per_job = nexec_per_job
        self.quiet = quiet
        self.prefix = prefix
        self.position = position
        self.allbar: Any = None
        self.compbar: Any = None
        self.evalbar: Any = None
        self.ntasks = 0
        self.progress = self._get_progress_cls(progress_cls)

    def _get_progress_cls(self, progress_cls: str):
        if progress_cls == "tqdm":
            import tqdm

            return tqdm.tqdm
        assert False, f"unknown progress class name: {progress_cls}"

    @override
    def search_start(self, ntasks: int):
        self.ntasks = ntasks
        tqdm_args = dict(
            total=self.ntasks,
            miniters=0,
            mininterval=0,
            smoothing=0,
            disable=self.quiet,
        )
        self.allbar = self.progress(
            desc=f"{self.prefix} evaluate".strip(),
            colour="red",
            position=self.position + 0,
            **tqdm_args,  # type: ignore
        )
        self.compbar = self.progress(
            desc=f"{self.prefix} compile".strip(),
            colour="blue",
            position=self.position + 1,
            **tqdm_args,  # type: ignore
        )
        if self.nexec_per_job > 0:
            self.evalbar = self.progress(
                desc=f"{self.prefix} execute".strip(),
                colour="green",
                position=self.position + 2,
                **tqdm_args,  # type: ignore
            )

    @override
    def batch_start(self, batch_ntasks: int):
        pass

    @override
    def compile_batch_start(self):
        self.compbar.unpause()

    @override
    def compile_job_start(self):
        pass

    @override
    def compile_job_end(self):
        self.compbar.update(self.ncomp_per_job)
        if self.nexec_per_job == 0:
            self.allbar.update(self.ncomp_per_job)

    @override
    def compile_batch_end(self):
        self.compbar.update(0)
        self.allbar.update(0)

    @override
    def execute_batch_start(self):
        if self.nexec_per_job > 0:
            self.evalbar.unpause()

    @override
    def execute_job_start(self):
        pass

    @override
    def execute_job_end(self):
        if self.nexec_per_job > 0:
            self.evalbar.update(self.nexec_per_job)
            self.allbar.update(self.nexec_per_job)

    @override
    def execute_batch_end(self):
        pass

    @override
    def batch_end(self):
        self.compbar.unpause()
        if self.nexec_per_job > 0:
            self.evalbar.unpause()

    @override
    def search_end(self):
        self.allbar.close()
        self.compbar.close()
        if self.nexec_per_job > 0:
            self.evalbar.close()


class SearchProgressMO(SearchProgress):
    def __init__(
        self,
        ncomp_per_job: int = 1,
        nexec_per_job: int = 1,
        quiet: bool = False,
        prefix: str = "",
        position: int = 0,
        progress_cls: str = "mo",
    ):
        self.ncomp_per_job = ncomp_per_job
        self.nexec_per_job = nexec_per_job
        self.quiet = quiet
        self.prefix = prefix
        self.allbar: Any = None
        self.ntasks = 0
        self.progress = self._get_progress_cls(progress_cls)
        self.ncomp = 0
        self.nexec = 0
        self._lock = threading.Lock()

    def _get_progress_cls(self, progress_cls: str):
        if progress_cls == "mo":
            import marimo

            return marimo.status.progress_bar
        assert False, f"unknown progress class name: {progress_cls}"

    @override
    def search_start(self, ntasks: int):
        self.ntasks = ntasks
        tqdm_args = dict(
            total=self.ntasks,
            remove_on_exit=False,
        )
        self.allbar = self.progress(
            subtitle="Starting...",
            completion_subtitle="Explored {self.ntasks} samples.",
            title=f"{self.prefix} evaluate".strip(),
            **tqdm_args,  # type: ignore
        ).progress

    @override
    def compile_job_end(self):
        self.ncomp += self.ncomp_per_job
        if self.nexec_per_job == 0:
            with self._lock:
                self.allbar.update_progress(
                    self.ncomp_per_job,
                    subtitle=f"Compiled {self.ncomp}/{self.ntasks}",
                )

    @override
    def execute_job_end(self):
        if self.nexec_per_job > 0:
            self.nexec += self.nexec_per_job
            with self._lock:
                self.allbar.update_progress(
                    self.nexec_per_job,
                    subtitle=f"Evaluated {self.nexec}/{self.ntasks}",
                )

    @override
    def search_end(self):
        self.allbar.close()
