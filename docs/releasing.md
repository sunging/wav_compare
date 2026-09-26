# Version and release maintenance

## Commits

Use Conventional Commit messages and PR titles. Squash merging makes the final PR title the release input:

```text
feat: add channel selection
fix: preserve PCM32 precision
perf: cache waveform envelopes
feat!: change report semantics
```

`fix`/`perf` bump patch, `feat` bumps minor. Breaking changes bump minor during 0.x and major after 1.0; explain migrations with a `BREAKING CHANGE:` footer. Documentation, tests and internal chores alone do not require a release. Use `fix(deps):` for dependency fixes requiring a patch release.

## Release PR

Main updates run release-please's Python strategy to maintain one release PR with package version, release manifest and generated English CHANGELOG. A follow-up runs `uv lock` on that branch to update the local package version without upgrading dependencies. The exact resulting SHA is passed to the reusable cross-platform test/build workflow. Bot-created PRs are not assumed to trigger normal CI with `GITHUB_TOKEN`.

The bootstrap version is 0.0.0; first feature commits prepare 0.1.0. The initial release PR stays open. Main is installable before formal release, and release changelog sections remain in that PR until merged.

Before merging, inspect **Release preparation → release-pr-checks**, confirm the checkout SHA equals the current PR head, and review version/changelog/lockfile. Checks for an earlier main commit do not prove release readiness. `uv run python tools/check_version.py` verifies package, manifest and locked local version agree.

Merging authorizes creation of `vX.Y.Z` and a GitHub Release. The same workflow checks out the release SHA, verifies its tag and CHANGELOG, builds wheel/sdist and uploads artifacts. It does not rely on generated tag events starting another workflow. Nothing is published to PyPI.

Use only the built-in repository token with contents/issues/pull-request write permissions. Enable **Allow GitHub Actions to create and approve pull requests** in Actions settings. Do not copy local gh credentials into secrets. Workflow actions are pinned to immutable commits.

## Recovery

Re-running preparation updates the existing release PR. Lock synchronization commits use `chore: synchronize release lockfile`; investigate any unexpected dependency upgrades. If artifact upload fails after a release was created, rebuild the exact tag, validate it with `tools/check_version.py --tag TAG`, then run `gh release upload TAG dist/* --clobber`. Do not move the tag.

Installed metadata supplies CLI and About versions; never add a duplicate version literal. CI checks package/manifest/lock agreement, and release validation also requires a matching CHANGELOG section.
