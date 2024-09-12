
[comment]: # (Logo and Title)
<h1 align="center">
    <img src="https://github.com/mscaudill/simplexers/blob/main/docs/imgs/logo.png" 
    style="width:600px;height:auto;"/>
</h1>

<h2 align="center">
  <i><font color='gray'>Euclidean Projections onto Positive and Capped Simplicess</font></i>
</h2>


[comment]: # (Badges)
<p align="center">
  <a href="https://github.com/mscaudill/simplexers/blob/main/LICENSE"><img
    src="https://img.shields.io/badge/License-BSD%203--Clause-teal" 
    alt="Simplexers is released under the BSD 3-Clause license." />
  </a>
  <a href="https://pypi.org/project/openseize/"><img 
    src="https://img.shields.io/pypi/v/openseize?color=78437E&logo=pypi&logoColor=white" 
    alt="Openseize pypi release" />
  </a>
  <a href="https://github.com/mscaudill/openseize/tree/master#Dependencies"><img 
    src="https://img.shields.io/pypi/pyversions/openseize?logo=python&logoColor=gold" 
    alt="Python versions supported." />
  </a>
  <a href="https://github.com/mscaudill/openseize/actions/workflows/test.yml"><img 
    src="https://img.shields.io/github/actions/workflow/status/mscaudill/simplexers/test.yml?label=CI&logo=github" 
    alt="Simplexers' test status" />
  </a>
 <a href="https://github.com/mscaudill/openseize/pulls"><img 
    src="https://img.shields.io/badge/PRs-welcome-F8A3A3"
    alt="Pull Request Welcomed!" />
  </a>
</p>


[comment]: # (Navigation links)
<p align="center"  style="font-size: 20px">
<a href="#Key-Features">About</a>   |  
<a href="#Installation">Installation</a>   |  
<a href="#Dependencies">Dependencies</a>   |  
<a href="#Attribution">Attribution</a>   |  
<a href="#Contributions">Contributions</a>   |  
<a href="#Issues">Issues</a>   |  
<a href="#License">License</a> |
<a href="#Acknowledgements">Acknowledgements</a> 
</p>

# About

The s-capped simplex is defined as:
```math
\Delta_{s}^{=} := \{\mathbf{x} \in \mathbb{R}^{n} \mid \mathbf{x}^T\mathbf{1} = s,
 \mathbf{0} \leq \mathbf{x} \leq \mathbf{1} \}
```

[comment]: # (simplex feasible region)
<h1 align="center">
    <img src="https://github.com/mscaudill/simplexers/blob/main/docs/imgs/simplex_region.png" 
    style="width:300px;height:auto;"/>
</h1>

Geometrically, the simplex is a slice at $\mathbf{x}^T\mathbf{1} = s$ in
a hypercube $\mathbf{0} \leq \mathbf{x} \leq \mathbf{1}$ shown by the blue
region in the image.

Projecting a vector onto a simplex is an important subproblem that appears in
imaging, statistics, and machine learning applications [1](https://link.springer.com/article/10.1007/s10107-015-0946-6) [2](https://proceedings.neurips.cc/paper/2021/file/52aaa62e71f829d41d74892a18a11d59-Paper.pdf) [3](https://www.sciencedirect.com/science/article/abs/pii/S0167865522002185).

The projection of vector $\mathbf{y}$ onto the simple amounts to finding a 
vector $\mathbf{x}*$ that lives in the blue (feasible) region. This vector is
the lower dimensional *shadow* of $\mathbf{y}$. Formally, this projection is
written as:
```math
\mathbf{x}* = proj_{\Delta_{s}^{=}}\left(\mathbf{y}\right) = \underset{x}{\mathrm{argmin}}\{
\frac{1}{2}\|\|\mathbf{x} - \mathbf{y}\|\|^2 \mid \mathbf{x} \in
\Delta_{s}^{=}\}
```

# Installation

Simplexers is available on <a href=https://pypi.org/project/simplexers/" 
                            target=_blank>pypi</a> for easy intallation into
virtual environments.

### Python Virtual Environment

1. Create your virtual environment, Here we name it `my_venv`. 
```Shell
$ python3 -m venv my_venv
```

2. Activate your 'my_venv' environment
```Shell
$ source my_venv/bin/activate
```

3. Install openseize into your virtual environment
```Shell
(my_venv)$ pip install openseize
```

### Conda Virtual Environment

1. Download the openseize environment <a
href=https://github.com/mscaudill/simplexers/blob/master/environment.yml 
target=_blank>configuration yaml</a> 


2. Create a conda simplexers environment.
```Shell
$ conda env create --file environment.yml
```

3. Activate the `simplexers` environment.
```Shell
$ conda activate openseize
```

4. Install simplexers from pypi into your simplexers environment.
```Shell
(openseize)$ pip install simplexers
```

### From Source

To get the development version:

1. Create a virtual environment with latest pip version.
```Shell
$ python3 -m venv env
$ source env/bin/activate
$ pip install --upgrade pip
```

2. Get the source code
```Shell
$ git clone https://github.com/mscaudill/simplexers.git
```

3. CD into the directory containing the pyproject.toml and create an 
editable install with `pip` using the development dependencies
```Shell
$ pip install -e .[dev]
```

# Dependencies

Simplexers requires <b>Python <span>&#8805;</span> 3.10</b> and has the
following dependencies:

<table>

  <tr>
    <th>package</th>
    <th>pypi</th>
    <th>conda</th>
  </tr>

  <tr>
    <td><a href="https://numpy.org/doc/stable/index.html#" 
        target=_blank>numpy</a></td>
    <td>https://pypi.org/project/numpy/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

  <tr>
    <td><a href="https://scipy.org/" 
        target=_blank>scipy</a></td>
    <td>https://pypi.org/project/scipy/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

  <tr>
    <td><a href="https://ipython.org/" 
        target=_blank>ipython</a></td>
    <td>https://pypi.org/project/ipython/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

  <tr>
    <td><a href=https://jupyter.org/ 
        target=_blank>notebook</a></td>
    <td>https://pypi.org/project/jupyter/</td>
    <td align='center'><span>&#10003;</span></td>
  </tr>

</table>

# Attribution

Please see the *Cite this repository* under the About section or the [citation
file](https://github.com/mscaudill/openseize/blob/master/CITATION.cff).


# Contributions

Contributions are what makes open-source fun and we would love for you to
contribute. Please check out our [contribution guide](
https://github.com/mscaudill/simplexers/blob/master/.github/CONTRIBUTING.md)
to get started.

# Issues

Simplexers provides custom issue templates for filing bugs, requesting
feature enhancements, suggesting documentation changes, or just asking
questions. *Ready to discuss?* File an issue <a
href=https://github.com/mscaudill/simplexers/issues/new/choose>here</a>. 

# License

Simplexers is licensed under the terms of the 3-Clause BSD License.

# Acknowledgements

**This work is generously supported through the Ting Tsung and Wei Fong Chao 
Foundation**

