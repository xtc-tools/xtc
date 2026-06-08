from types import SimpleNamespace as NS
from xtc.search.pipeline import CompileExecutePipeline
import time
import pytest

def _delay():
    time.sleep(0.1)

def _compile(payload: NS):
    print(f"compile {payload.id}")
    assert payload.input == f"Input {payload.id}"
    _delay()
    return NS(id=payload.id, compiled=f"Compiled {payload.id}")

def _execute(payload: NS):
    print(f"execute {payload.id}")
    assert payload.compiled == f"Compiled {payload.id}"
    time.sleep(0.1)
    _delay()
    return NS(id=payload.id, executed=f"Executed {payload.id}")

@pytest.mark.parametrize(
    "compile_jobs, execute_jobs, ntasks",
    [
        (1, 1, 4),
        (0, 0, 4),
        (3, 4, 11),
    ],
)
def test_pipeline(compile_jobs: int, execute_jobs: int, ntasks: int):
    pipeline = CompileExecutePipeline(
        compile_jobs = compile_jobs,
        execute_jobs = execute_jobs,
        compile_func=_compile,
        execute_func=_execute,
    )
    payloads = (NS(id=id, input=f"Input {id}") for id in range(ntasks))
    tasks = pipeline.run(payloads)
    for task in tasks:
        print(task)
    assert all([task.id == id for id, task in enumerate(tasks)])
    assert all([task.comp_result.compiled == f"Compiled {id}" for id, task in enumerate(tasks)])
    assert all([task.exec_result.executed == f"Executed {id}" for id, task in enumerate(tasks)])
