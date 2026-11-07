# Contributing

Thank you for your interest! Bug reports, ideas and pull requests are welcome.

## Reporting a problem

Please open an issue with:

- what you did and what happened (error message, serial log of the board, output of the script),
- firmware version (tab Info on the board) and ESP-IDF version, or the trainer page and Python version,
- your board and camera sensor if it is about the firmware.

**Do not post** your Anthropic API key, Wi-Fi passwords or photos you do not want to publish.
`~/.pokerdice/ai_last_answer.txt` contains the last AI answer without the key, if that helps.

## Pull requests

- One topic per pull request, with a short description of what and why.
- Code, comments, UI texts and documentation in English.
- Python: scripts with `argparse`, `--help` and example calls in the header; no fixed paths - everything relative to
  the repository or given as an argument; existing files are never overwritten (back up or refuse instead).
- Firmware: builds with ESP-IDF 6.x without new warnings; secrets only in NVS, never in the code.
- If you change the pre- or post-processing, change `firmware/main/preprocess.hpp` and the Python side
  (`trainer/colab/scripts/ml_tools.py`, `trainer/convert/decode.py`) together and run
  `firmware/tools/test_pc/compare.py`.
- If you change the Colab notebooks, change `trainer/colab/build_notebooks.py` (not the .ipynb files), raise the
  notebook version there and run it with `--force`.
- Web page changes can be tried on the PC: `python firmware/tools/web_preview.py --play`.
- Add a line to `CHANGELOG.md`.

## License

By contributing you agree that your contribution is published under the
[GNU Affero General Public License v3.0](LICENSE) like the rest of the project.
