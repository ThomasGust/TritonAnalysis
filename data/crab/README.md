# Crab Reference Data

`templates/` contains the crab reference images needed by the crab counter and
the synthetic crab dataset tools. Keep small, shared species-reference assets
here so another TritonAnalysis checkout can build the same reference atlas
without depending on a local `Downloads` folder or ignored `Workspace/data`
content.

Image filenames should include one of:

- `green` or `euro` for European green crab
- `rock` or `native` for native rock crab
- `jonah` for Jonah crab

Large run captures, generated datasets, and result folders should stay under
`Workspace/` and remain outside git.
