# Getting started

There are two ways to use this project:

- **A - only the firmware** with the example models in `firmware/models/`: you need ESP-IDF and the board.
  See [firmware.md](firmware.md).
- **B - your own dice**: you also need the trainer (Python), a Google account for Colab and some time for photos.
  This page covers the installation of the trainer.

## What you need for B

| What | Why |
|---|---|
| The camera board with the Pokerdice firmware | live images, single images and real test throws for the trainer |
| Python **3.11** | the converter esp-ppq only installs on 3.11; it may live next to other Python versions |
| Google account, Google Drive, Colab | rendering (Blender) and training run on a Colab GPU; a free account works, a paid plan is faster |
| Anthropic API key (optional) | AI helpers in the trainer: assessing the camera setup, reading the die faces, suggesting the die layout and labels |
| ESP-IDF 6.x | building and flashing the firmware |

## Installation

Windows (PowerShell, in the repository folder):

```
powershell -ExecutionPolicy Bypass -File setup.ps1
start.bat
```

Linux / macOS:

```
./setup.sh
./start.sh
```

The setup script creates the Python environment `.venv` in the repository folder and installs everything from
`trainer/requirements.txt` (torch in the CPU version first). Running it again only updates the packages.
`start.bat` / `start.sh` start the trainer and open http://127.0.0.1:5000 in the browser.

Start options (other port, other workspace, no browser):

```
.venv\Scripts\python.exe trainer\assistant\app.py --help        (Windows)
.venv/bin/python trainer/assistant/app.py --help                (Linux/macOS)
```

## First start

1. **0 · Setup** in the trainer: the address of the camera board (e.g. `pokerdice.local` or its IP address),
   the folder name on Google Drive, optionally the Anthropic API key and the model.
2. **New project**: one project per kind of dice (and tray). The folder name may contain letters, digits, `-` and `_`.
3. Then the steps of the trainer one after the other - see [trainer.md](trainer.md).

## Where your data lives

| Place | Content |
|---|---|
| `workspace/projects/<project>/` | `project.json`, photos, camera setup, test images, `export/` (from Colab), `models/` (converted), evaluations |
| `workspace/colab_packages/` | the ZIP packages for Colab and the matching first notebook |
| `workspace/firmware_models_backup/` | what was in `firmware/models/` before `select_project.py` replaced it |
| `~/.pokerdice/` (Windows: `%USERPROFILE%\.pokerdice\`) | settings of the trainer and the API key (`api_key.txt`) |
| your Google Drive, folder from step 0 | packages, scripts, notebooks, renders, training runs, `export/` |

`workspace/` is ignored by git: your photos never end up in the repository.
The API key can also come from the environment variable `ANTHROPIC_API_KEY`; it is never written into the repository,
a Colab package or the firmware.

## Common problems

| Message | Solution |
|---|---|
| "No suitable Python runtime found" for `py -3.11` | install Python 3.11 from python.org; installed versions: `py -0` |
| "running scripts is disabled" | start the script with `-ExecutionPolicy Bypass` as shown above |
| "No module named 'flask'" | the trainer was started with another Python - use `start.bat` / `start.sh` |
| port 5000 in use | `start.bat --port 5050` |
| the trainer does not find the board | open `http://<address>/api/info` in the browser; `.local` names do not work in every network - use the IP address |
