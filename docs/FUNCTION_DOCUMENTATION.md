# Function documentation standard

The owner must be able to understand and call each project function without reverse-engineering its body. Use Google-style Python docstrings and type hints. This project adopts the section format from the [Google Python style guide](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) and the general conventions in [PEP 257](https://peps.python.org/pep-0257/); the coverage and teaching requirements below are this project's choices.

## Coverage

Document every project-owned function and method, including private/nested helpers, constructors and CLI entry points. Cover new functions as they are written and existing functions as their module is worked on. When a change is finished, check that every function it touched is documented. A test function needs a brief explanation of the behavior and important setup; add detailed sections only when useful.

Keep the pinned upstream checkout unchanged. Explain upstream calls in the project's wrapper docstrings, and link to the pinned upstream source. Do not copy every upstream implementation into project documentation.

## What each function explains

Start with a short purpose statement: what the function does and why a caller would use it. Write the docstring immediately inside the function, using triple double quotes. Put types on the signature where practical; describe meaning rather than simply repeating a type name.

Use the following sections when applicable:

| Section | Required content for this project |
|---|---|
| `Args:` | Every caller-supplied argument by its exact name: meaning, allowed values, relevant units, constraints, and what an optional/default value means. Do not list implicit `self`/`cls`. Explain `*args` and `**kwargs` when used. |
| `Returns:` | What the result means and how to use it. Describe mapping keys, tuple order, nullable cases, exit codes and important invariants. For a function returning only `None`, state that it returns nothing and describe its effect; a separate section is optional. |
| `Yields:` | For a generator, describe each emitted item and ordering; explain exhaustion/resource cleanup when relevant. |
| `Raises:` | Exceptions callers should handle, with the conditions causing them. Distinguish returned validation errors from raised exceptions. |
| `Notes:` | Relevant side effects and prerequisites: mutated inputs, files read/written, network calls, environment variables, randomness, model loading, memory/compute cost or required reviewed artifacts. Say when a function is pure. |
| `Examples:` | At least one typical call and the expected result or artifact. Show the actual import or how the caller obtains the object. Use safe small inputs; distinguish illustrative fixtures from real measurements. |

Avoid empty headings and boilerplate. A tiny helper may explain its arguments, return and example concisely in prose. The information is mandatory; long sections are not. Explain tricky internal decisions in nearby comments, while the docstring describes the caller's contract.

For this audio/text project, be precise about seconds versus samples, Hz, bytes versus MiB, tensor shape/dtype/device, Unicode codepoint offsets versus clusters, half-open intervals, repository-relative paths, hashing policy and split/manifest eligibility. Specify only facts relevant to that function, verified against its implementation.

## Calling conventions

Show keyword arguments for meaningful options when the signature allows them; preserve positional-only parameters and existing APIs. Explain optional defaults and keyword-only parameters. Show methods on an initialized object, asynchronous calls with `await`, and iteration for generators. For CLI entry points, show the actual shell command, required working directory/interpreter, output location and exit-code meanings. Do not encourage calling a CLI `main()` as if it were a reusable library API.

Examples must use real function names, valid inputs and actual defaults. Do not invent an import, API, output value or successful model result. Keep side-effecting examples out of automatically executed doctests unless explicitly isolated and authorized. Check ordinary pure examples with small local calls; use a clearly labeled illustrative example when resources prevent execution.

## Example using an existing project function

Below is a documentation/type-hint illustration of `khmer_tts.normalize.normalize`. Its implementation body is omitted **only in this guide**; this is not a new function or a runnable replacement.

```python
def normalize(text: str) -> dict[str, object]:
    """Prepare written transcript text and flag forms needing review.

    Args:
        text: Original Unicode transcript. Digits and Latin text are retained,
            not expanded into an assumed pronunciation.

    Returns:
        A mapping with:
            text_training: Text after the versioned normalization rules.
            changes: Rule names mapped to the counts of changes made.
            flags: Sorted review-category names, including an empty-text flag
                when normalization leaves no text. This is not QC approval.

    Raises:
        TypeError: If text is not a string.

    Notes:
        Pure: does not mutate inputs, write files or access the network.
        Applying normalization again preserves the normalized text; the
        change log describes the current call, not cumulative history.

    Examples:
        >>> from khmer_tts.normalize import normalize
        >>> result = normalize(text="  កម្ពុជា  ")
        >>> result["text_training"]
        'កម្ពុជា'
    """
    ...
```

To call the actual existing function safely from the repository root:

```bash
PYTHONPATH=src python3 -c 'from khmer_tts.normalize import normalize; print(normalize(text="  កម្ពុជា  ")["text_training"])'
```

Expected output: `កម្ពុជា`. This example validates a text-normalization call; it makes no speech-quality claim.
