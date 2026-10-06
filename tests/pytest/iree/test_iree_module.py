from xtc.targets.iree.IREEModule import IREEModule


def _spec():
    return [{"shape": (4, 4), "dtype": "float32"}]


def _module():
    return IREEModule(
        name="m",
        payload_name="m",
        file_name="/nonexistent/m.vmfb",
        graph=None,
        np_inputs_spec=_spec,
        np_outputs_spec=_spec,
        reference_impl=lambda *a: None,
    )


def test_module_uses_explicit_specs_without_graph():
    module = _module()
    assert module._np_inputs_spec is _spec
    assert module._np_outputs_spec is _spec
    assert module._reference_impl is not None
    assert module.file_type == "vmfb"


def test_evaluator_and_executor_expose_module():
    module = _module()
    assert module.get_evaluator().module is module
    assert module.get_executor().module is module
