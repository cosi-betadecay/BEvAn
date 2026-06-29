---
name: no-underscore-prefixed-names
description: Code convention — never use leading-underscore (_private) function/symbol names in this repo
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ece69588-95fa-4489-9127-cfb3217fee8e
---

Never introduce leading-underscore names (`_unit`, `_ckd_residual`, etc.) for functions or module symbols in BEvAn. Use plain public names even for internal helpers.

**Why:** user explicitly rejected the `_`-prefix "private helper" convention ("Don't have `_FUNCTION` ever").

**How to apply:** when extracting nested helpers to module level or reviewing code, do not recommend or apply `_`-prefixing to signal internal-only use. Module-level helpers keep plain names. Relates to [[cosi-working-style]].
