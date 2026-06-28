# Installing, building & environment

## One-shot install (downloads & builds ROOT + Geant4 + MEGAlib)

```bash
bash ./setup.sh [options]
```

`setup.sh` (top of repo) clones/updates MEGAlib, downloads and builds compatible ROOT and
Geant4 (unless pointed at existing installs), applies patches from `resource/patches/`,
runs `./configure`, and compiles everything. Key flags (confirm with `bash setup.sh --help`):

| Flag | Meaning |
|------|---------|
| `--megalibpath=<path>` | where to install MEGAlib (default `./MEGAlib`) |
| `--externalpath=<path>` | where ROOT/Geant4 get installed (default `<megalibpath>/external`) |
| `--branch=<git-branch>` | git branch (default: latest release) |
| `--root=<path>` | use an existing ROOT (empty ⇒ download/build) |
| `--geant4=<path>` | use an existing Geant4 (empty ⇒ download/build) |
| `--heasoft=<path\|off>` | optional cfitsio/HEASoft (off by default) |
| `--maxthreads=<N>` | parallel build jobs (default: core count) |
| `--optimization=off\|normal\|strong` | compiler optimization (default normal) |
| `--debug=off\|on\|strong` | debug symbols / sanitizers |
| `--keepmegalibasis=on` | don't update the source, just recompile |
| `--patch=yes\|no` | apply MEGAlib's ROOT/Geant4 patches (default yes) |
| `--cleanup=on` | delete intermediate build files after |

The exact flag names matter and have varied across versions — **read the `confhelp()` /
option-parsing block in `setup.sh`** before scripting it.

## Environment setup (every shell, before using any tool)

`setup.sh` generates a source script (commonly `bin/source-megalib.sh`, on some setups
named via `--keepenvironmentasis`/`env.sh`). Source it:

```bash
source ~/MEGAlib/bin/source-megalib.sh     # sets $MEGALIB, PATH, LD_LIBRARY_PATH, PYTHONPATH
```

Make it permanent in `~/.bashrc` / `~/.zprofile`:

```bash
. $HOME/MEGAlib/bin/source-megalib.sh
```

Verify: `echo $MEGALIB` should print the install root; `which cosima` should resolve.
Inspect dependency versions with `megalib-config --root` / `--geant4` / `--compiler`.

> If a tool prints `error while loading shared libraries: libMEGAlib.so` or a ROOT/Geant4
> lib, the environment was not sourced (or `LD_LIBRARY_PATH` is wrong). This is the #1
> "it doesn't run" cause.

## Rebuilding after editing source

You do **not** re-run `setup.sh` to recompile changed code. From `$MEGALIB`:

```bash
make -j$(nproc)        # incremental build of everything
make clean             # remove all build artifacts (lib/, include/ links)
```

`./configure` only needs re-running to change architecture/optimization/debug/compiler:

```bash
./configure --architecture=linux --optimization=normal --debug=off --compiler=gcc
```

It writes `config/Makefile.options` + `config/Makefile.config` (selected from
`config/Makefile.{linuxgcc,linuxicc,linuxclang,macosx}`); `config/Makefile.user` is for
your own overrides and is preserved.

### Per-module targets (faster iteration)

The top-level `Makefile` routes short targets to subdirectories. Approximate map (verify
against the `Makefile`):

| Target | Module |
|--------|--------|
| `make glo` | global libraries |
| `make geo` | geomega |
| `make cos` | cosima |
| `make rev` | revan |
| `make mim` | mimrec |
| `make siv` | sivan |
| `make res` | response |
| `make fre` | fretalon |
| `make spe` | spectralyze |
| `make rea` | realta |
| `make evi` | eview |
| `make add` | addon tools |

After building module objects, the build **links** them (each module's `link` target
appends to `lib/AllObjects.txt`) and `make combine` produces the unified `libMEGAlib.so`.

### The ROOT dictionary step (important when adding/renaming classes)

Each module generates a ROOT dictionary so classes are serializable and visible to PyROOT:

```
bin/generatelinkdef -o <Lib>_Dictionary_LinkDef.h -i <headers>   # builds the LinkDef
rootcling -f <Lib>_Dictionary.cxx -I include -D___CLING___ \
          -rmf lib<Lib>.rootmap -s lib<Lib> -c <headers> <LinkDef>
```

Practical consequence: a class needs `ClassDef(MName, <version>)` in the header and
`ClassImp(MName)` in the `.cxx` to participate. Adding a new header to a module's `Makefile`
`FILES`/`HEADERS` list is required for it to compile and get a dictionary entry. See
`conventions-contributing.md`.

## Version requirements

Build-time gating files in `config/`:
- `AllowedROOTVersions.txt` — accepted ROOT versions (e.g. `632 636` ⇒ ROOT 6.32.x / 6.36.x).
- `AllowedGeant4Versions.txt` — accepted Geant4 versions.
- `AllowedHEASoftVersions.txt` — accepted HEASoft (optional).

**Always read these files for the current truth** — they change over time, and a
"version not allowed" failure from `configure_rootversiontest`/`configure_geant4versiontest`
means your ROOT/Geant4 is outside the listed set. `resource/patches/` holds per-version
patches that `setup.sh --patch` applies.

## Docker

```bash
docker build -t megalib - < Dockerfile          # Ubuntu 22.04, builds full stack
docker run -it -e USERID=$(id -u) -e GROUPID=$(id -g) \
  -v $PWD/data:/home/mrmegalib/data megalib
```

The image creates a non-root `mrmegalib` user, builds via `setup.sh`, and pre-sources the
environment in `.bashrc`. It maps host UID/GID for shared volumes.

## Build troubleshooting checklist

1. `echo $MEGALIB` set? Environment sourced? (most failures)
2. ROOT/Geant4 versions inside the Allowed*.txt sets? (`megalib-config --root`/`--geant4`)
3. After changing headers/class names: did you re-`make` the module so the dictionary
   regenerates? Stale `lib/*_Dictionary.cxx` / `.rootmap` cause cryptic link/load errors.
4. `make clean && make -j` resolves most stale-object problems.
5. macOS: ensure `DYLD_LIBRARY_PATH`/`LD_LIBRARY_PATH` from the source script are present;
   Geant4 data env vars (`G4*DATA`) must point at installed datasets for cosima.
