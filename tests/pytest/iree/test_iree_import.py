def test_iree_packages_importable():
    """The IREE backend and target packages import without the IREE runtime.

    At this stage both packages are empty scaffolding: importing them must not
    require ``iree-base-compiler`` / ``iree-base-runtime`` to be installed.
    """
    import xtc.backends.iree as backend
    import xtc.targets.iree as target

    assert backend.__all__ == []
    assert target.__all__ == []
