Some examples for running different types of atmospheres can be found here,
including the ones shown in Tsai+ 2021.

The .py files in this directory are the legacy flat-globals configs. As of
the TOML/Pydantic migration, neoVULCAN uses vulcan_cfg.toml (loaded into a
typed `VulcanConfig` schema). To use one of these examples:

  1. Copy the example .py file's values into the neoVULCAN/vulcan_cfg.toml
     structure (see the live vulcan_cfg.toml for the section layout).
  2. Or write the example as a standalone TOML and run side-by-side with:
        python vulcan.py -c cfg_examples/my_example.toml

The .py files are kept for reference (and so the test_regression.py
ORIG_DIR comparison against ../vulcan/ still has matching configs).
