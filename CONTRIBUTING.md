# Contributing

Nocturne keeps Forge Neo integration separate from fork development.

- `origin` points to the Nocturne fork.
- `upstream-neo` points to the Forge Classic Neo repository and is treated as read-only.
- `main` contains Nocturne development.
- `integration/upstream-neo` is reserved for reviewing upstream updates before they reach `main`.

Fetch upstream changes with:

```text
git fetch upstream-neo neo
```

Do not push branches or tags to `upstream-neo`.
