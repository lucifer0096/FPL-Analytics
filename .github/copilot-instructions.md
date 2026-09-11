# FPL Analytics Agent Guidelines

## Documentation

- When a change affects setup, usage, architecture, model behavior, deployment, workflows, or user-visible features, update `README.md` in the same change when the existing documentation needs correction or addition.
- Do not add README churn for internal refactors that do not change documented behavior.

## Validation

- Run the narrowest relevant executable check after each edit, then run the broader available checks before publishing.
- Do not claim deployment or test success without command output.
- Inspect `git diff --check` and `git status` before publishing.

## Publishing Changes

- For agent-created changes, after validation succeeds, commit the complete related change set and push it to the current branch's configured remote.
- Use a concise commit message that describes the user-visible or technical change.
- Never force-push, reset, rewrite history, create a branch, or commit unrelated user changes.
- Never commit secrets, credentials, `.env` contents, personal manager snapshots, generated caches, or other ignored private data.
- If validation fails, the worktree contains unrelated changes, or the remote rejects the push, stop before committing/pushing and report the concrete blocker.
- After pushing, verify that the local branch is aligned with its remote and report the commit hash.
