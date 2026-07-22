def test_iree_packages_importable():
    """The IREE backend and target packages import without the IREE runtime.

    Importing them (and the backend entry point) must not require
    ``iree-base-compiler`` / ``iree-base-runtime`` to be installed: only
    compilation and execution do.
    """
    import xtc.backends.iree as backend
    import xtc.targets.iree as target

    assert "Backend" in backend.__all__
    assert backend.Backend is backend.IREEBackend
    assert target.__all__ == []
