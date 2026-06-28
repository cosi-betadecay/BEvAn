# Coding conventions & contributing

Authoritative sources in the repo: `CodingConventions.md`, `.clang-format`,
`Contribute.md`, `CodeOfConduct.md`. Match the surrounding file; MEGAlib is internally very
consistent. Mirror existing idioms over "modernizing".

## Naming
- **Classes**: start with `M` → `MVector`, `MImage`. GUI option/expo classes: `MGUIOptions`,
  `MGUIExpo`. Modules: `MModule...`. (Cosima uses `MC...` with `.hh/.cc`.)
- **Methods**: UpperCamelCase verbs — `SetMagThetaPhi`, `IsNull`, `Clear`, `GetEnergy`.
- **Member variables**: `m_UpperCamelCase` — `m_X`, `m_DataPoint`.
- **Local variables**: UpperCamelCase — `X`, `DataPoint` (no `m_`).
- **Constants/statics**: `c_Name` / `g_Name` for class-constants / globals.

## Formatting (`.clang-format`)
- Base LLVM, C++ (project targets C++14/17 per platform Makefiles). **2-space indent, no
  tabs.** Pointer binds left (`int* p`). No hard column limit. Includes not reordered.
- Braces: opening brace on a new line for **functions/classes/methods**; at end-of-line for
  control statements (`if (...) {`). Space after `if/for/while/switch`.
- Big separator comment banners between functions:
  `////////////////////////////////////////////////////////////////////////////////`.
- Run `clang-format` (config is in repo root) before committing C++ changes.

## Comments
- Doxygen `//!` for classes, methods, and members (drives the generated docs / Doxyfile).
- Plain `//` for inline reasoning. Explain **why**, not what.

## C++ practices the project asks for
- Meaningful names; `const`/`constexpr`; no magic numbers; always initialize variables.
- Prefer `std::vector` over C arrays. Use the `MExceptions` types for error signaling.
- Use the `MStreams` logging macros (`mout/merr/mlog/mdebug/massert`) — not raw iostream.
- Avoid new global variables.
- NOTE the reality gap: existing code uses **manual `new`/`delete`, raw pointers, no smart
  pointers**, and `using namespace std;` in headers. New code should be cleaner where it can,
  but must respect existing ownership conventions (callers delete returned events/objects) —
  don't introduce smart pointers into APIs that hand ownership across the manual boundary
  without checking call sites.

## Adding a class (and its ROOT dictionary)
1. Create `src/<module>/inc/MFoo.h` and `src/<module>/src/MFoo.cxx`.
2. Header: include guard, `//!` docs, `ClassDef(MFoo, <version>)` at the end of the class
   (needed for ROOT I/O + PyROOT visibility).
3. Source: `ClassImp(MFoo)` near the top.
4. Add `MFoo` to that module's `Makefile` (`FILES`/`HEADERS` list) so it compiles and is
   picked up by `generatelinkdef` + `rootcling`.
5. `make <moduletarget>` — the dictionary regenerates. If you rename/remove a class, do
   `make clean` for that module to clear stale `*_Dictionary.cxx`/`.rootmap`/`.pcm`.

See `install-build.md` for the dictionary mechanics and per-module make targets.

## Tests
- Such tests as exist live in `resource/examples/unittests/` (activations, shells,
  guardrings, cosima checksums) and a few module `unittests/` dirs (e.g.
  `src/cosima/unittests/UTCosimaInputSpectra.cc`, `src/mimrec/unittests/`). Run via the
  example's `run.sh`. There is **no CI build/test** — only CodeQL static analysis
  (`.github/workflows/codeql-analysis.yml`). Validate changes by re-running the relevant
  worked example end-to-end (see `workflows.md`) and comparing against reference outputs.

## Contributing flow
- See `Contribute.md`. Standard fork/branch/PR to the upstream `zoglauer/megalib` repo;
  match conventions, run clang-format, ensure the relevant example still produces sane
  output, and keep changes focused. The git commit log uses prefixes like `BUG:`, `CHG:`,
  `ADD:` (see `git log`).
