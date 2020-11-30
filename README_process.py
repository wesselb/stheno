examples = [
    {
        "title": "Simple Regression",
        "toc": "simple-regression",
        "file": "readme_example1_simple_regression",
    },
    {
        "title": "Decomposition of Prediction",
        "toc": "decomposition-of-prediction",
        "file": "readme_example2_decomposition",
    },
    {
        "title": "Learn a Function, Incorporating Prior Knowledge About Its Form",
        "toc": "learn-a-function-incorporating-prior-knowledge-about-its-form",
        "file": "readme_example3_parametric",
    },
    {
        "title": "Multi-Output Regression",
        "toc": "multi-output-regression",
        "file": "readme_example4_multi-output",
    },
    {
        "title": "Approximate Integration",
        "toc": "approximate-integration",
        "file": "readme_example5_integration",
    },
    {
        "title": "Bayesian Linear Regression",
        "toc": "bayesian-linear-regression",
        "file": "readme_example6_blr",
    },
    {"title": "GPAR", "toc": "gpar", "file": "readme_example7_gpar"},
    {
        "title": "A GP-RNN Model",
        "toc": "a-gp-rnn-model",
        "file": "readme_example8_gp-rnn",
    },
    {
        "title": "Approximate Multiplication Between GPs",
        "toc": "approximate-multiplication-between-gps",
        "file": "readme_example9_product",
    },
    {
        "title": "Sparse Regression",
        "toc": "sparse-regression",
        "file": "readme_example10_sparse",
    },
    {
        "title": "Smoothing with Nonparametric Basis Functions",
        "toc": "smoothing-with-nonparametric-basis-functions",
        "file": "readme_example11_nonparametric_basis",
    },
]

example_template = """
### {title}

![Prediction](https://raw.githubusercontent.com/wesselb/stheno/master/{file}.png)

```python
{source}
```
"""
toc_template = "    - [{title}](#{toc})"

# Fill the template.
out = ""
for example in examples:
    with open(example["file"] + ".py", "r") as f:
        source = f.read()
    out += example_template.format(
        title=example["title"], file=example["file"], source=source.strip()
    )

# Construct the ToC.
toc = "\n".join(
    [
        toc_template.format(title=example["title"], toc=example["toc"])
        for example in examples
    ]
)

# Read and fill README.
with open("README_without_examples.md", "r") as f:
    readme = f.read()
readme = readme.replace("{examples_toc}", toc)
readme = readme.replace("{examples}", out)

# Write result.
with open("README.md", "w") as f:
    f.write(readme)
