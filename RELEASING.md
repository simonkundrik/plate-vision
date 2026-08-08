# Releasing

Two things ship from this repository on separate schedules, because they change for
unrelated reasons and pinning them together would force a version bump on one every time
the other moved.

| What | Tag | Where it goes |
|---|---|---|
| `@plate-vision/client` | `client-v0.1.0` | npm |
| Model artifact + manifest | `model-v0.1.0` | GitHub Releases |

## The client package

```bash
npm version patch --workspace packages/client   # or minor / major
git push && git push --tags
```

Pushing a `client-v*` tag runs `.github/workflows/release-client.yml`, which typechecks,
lints, tests, checks the tag against `package.json`, and publishes with npm provenance.

The version check is not ceremony. A tag reading `client-v0.2.0` that publishes `0.1.0`
produces a release nobody can find and a tag that describes nothing.

**Setup required once:** an `NPM_TOKEN` secret with publish rights. On npm, prefer a
granular access token scoped to this package over a classic automation token.

## The model artifact

Model weights are tens of megabytes and cannot live in the repository, so a GitHub Release
is the distribution channel.

```bash
cd model
python scripts/export_model.py --classifier runs/kaggle-baseline/runs/baseline/best.pt \
                               --out runs/release --skip-quantization
python scripts/publish_model.py --export-dir runs/release --tag model-v0.1.0 --dry-run
python scripts/publish_model.py --export-dir runs/release --tag model-v0.1.0
```

`publish_model.py` refuses to publish unless the manifest describes the artifact sitting
next to it, byte count and SHA-256 both. That check exists because a mismatched pair is a
failure nobody notices at upload time: every client rejects the download, and the error
points at the network rather than at whoever uploaded the wrong file.

It also writes the licence position into the release notes every time, rather than relying
on whoever runs it to remember.

### Why the app pins a tag

`app/app.json` names a specific release tag, not `latest`. A build is tested against one
model; letting the artifact change underneath it means a build that passed verification and
a build in a user's hands can be running different models with no way to tell them apart.

Moving the app to a new model is a config change, reviewed like any other.

## Version relationships

The package version tracks the **API**. The model version tracks the **weights**. They move
independently and neither implies the other.

What connects them is `schema_version` in `bundle.json`. `parseBundle` refuses a version it
does not understand rather than reading fields that may have moved, so an old client and a
new artifact fail loudly instead of silently misinterpreting each other.

## Licence, every time

The code is MIT. **The published weights are not.** They are trained on Food-101, whose
images come from Foodspotting and are not ETH Zurich's to relicense; use beyond scientific
fair use must be negotiated with the individual image owners.

Every model release says so in its notes. Do not remove that from `publish_model.py`.
