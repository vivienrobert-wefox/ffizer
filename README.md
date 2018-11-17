# ffizer

<!-- copy badges from:
- [repostatus.org](https://www.repostatus.org/#active)
- [Shields.io: Quality metadata badges for open source projects](https://shields.io/#/)
-->

[![Crates.io](https://img.shields.io/crates/l/ffizer.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Crates.io](https://img.shields.io/crates/v/ffizer.svg)](https://crates.io/crates/ffizer)

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.com/davidB/ffizer.svg?branch=master)](https://travis-ci.com/davidB/ffizer)

[![Crates.io](https://img.shields.io/crates/d/ffizer.svg)](https://crates.io/crates/ffizer)
![GitHub All Releases](https://img.shields.io/github/downloads/davidB/ffizer/total.svg)

ffizer is a files and folders initializer / generator. Create any kind (or part) of project from template.

keywords: file generator, project template, project scaffolding, quickstart, project initializer, project skeleton

<!-- TOC -->

- [Motivations](#motivations)
    - [Main features](#main-features)
    - [Sub features](#sub-features)
- [Limitations](#limitations)
- [Usages](#usages)
    - [Install](#install)
        - [via github releases](#via-github-releases)
        - [via cargo](#via-cargo)
    - [Run](#run)
    - [Create your first template](#create-your-first-template)
- [Build](#build)
- [Alternatives](#alternatives)
    - [Generic](#generic)
    - [Specialized](#specialized)

<!-- /TOC -->

<a id="markdown-motivations" name="motivations"></a>
## Motivations

<a id="markdown-main-features" name="main-features"></a>
### Main features

- [X] project generator as a standalone executable (no shared/system dependencies (so no python + pip + ...))
- [X] a simple and generic project template (no specialisation to one ecosystem)
- [ ] template as simple as possible, like a
  - [ ] copy or clone with file/folder renames without overwrite
  - [X] few search and replace into file
- [X] template hosted as a local folder on the file system
- [ ] template hosted as a git repository on any host (not only public github)
  - [ ] at root of the repository
  - [ ] in subfolder of the repository
  - [ ] in any revision (branch, tag, commit)
- [X] a fast enough project generator

<a id="markdown-sub-features" name="sub-features"></a>
### Sub features

- [X] dry mode (usefull to test)
- [ ] chain template generation because fragment of templates can be commons
- [ ] chain commands (eg: 'git init') (like a post-hook)
  - [ ] raw command
  - [ ] template command

<a id="markdown-limitations" name="limitations"></a>
## Limitations

Some of the following limitations could change in the future (depends on gain/loss):

- no conditionnals file or folder creation
- no update of existing file or folder
- no specials features
- no plugin and not extensible (without change the code)
- handlebars is the only template language supported (support for other is welcome)

<a id="markdown-usages" name="usages"></a>
## Usages

<a id="markdown-install" name="install"></a>
### Install

<a id="markdown-via-github-releases" name="via-github-releases"></a>
#### via github releases

Download the binary for your platform from [github releases](https://github.com/davidB/ffizer/releases), then unarchive it and place it your PATH.

<a id="markdown-via-cargo" name="via-cargo"></a>
#### via cargo

```sh
cargo install ffizer
```

<a id="markdown-run" name="run"></a>
### Run

```txt
ffizer 0.3.0
davidB
ffizer is a files and folders initializer / generator. Create any kind (or part) of project from template.

USAGE:
    ffizer [FLAGS] [OPTIONS] --destination <dst_folder> --source <src_uri>

FLAGS:
    -h, --help                      Prints help information
    -V, --version                   Prints version information
    -v, --verbose                   Verbose mode (-v, -vv (very verbose / level debug), -vvv) print on stderr
        --x-always_default_value    should not ask for valiables values, always use defautl value or empty (experimental
                                    - for test only)

OPTIONS:
        --confirm <confirm>           ask confirmation 'never', 'always' or 'auto' (default) [default: auto]
    -d, --destination <dst_folder>    destination folder (created if doesn't exist)
    -s, --source <src_uri>            uri / path of the template
```

<a id="markdown-create-your-first-template" name="create-your-first-template"></a>
### Create your first template

( from scratch without ffizer ;-) )

```sh
# create the folder with the template
mkdir my-template
cd my-template

# add file that will be copied as is
cat > file0.txt <<EOF
I'm file0.
EOF

# add a template file that will be "rendered" by the handlebars engine
# - the file should have the .ffizer.hbs extension,
# - the extension .ffizer.hbs is removed from the generated filename
# - [Handlebars templating language](https://handlebarsjs.com/)
cat > file1.txt.ffizer.hbs <<EOF
I'm file1.txt of {{ project }}.
EOF

# add a file with a name that will be "rendered" by the handlebars engine
# - the file should have {{ variable }},
# - [Handlebars templating language](https://handlebarsjs.com/)
cat > '{{ project }}.txt' <<EOF
I'm a fixed content file with rendered file name.
EOF

```

- The minimal template is an empty dir.
- a sample template and its expected output (on empty folder) is available at [tests/test_1](tests/test_1).
- file priority (what file will be used if they have the same destination path)

  ```txt
  existing file
  file with source extension .ffizer.hbs (and no {{...}} in the source file path)
  file with identical source file name (and extension)
  file with {{...}} in the source file path
  ```

<a id="markdown-build" name="build"></a>
## Build

<a id="markdown-alternatives" name="alternatives"></a>
## Alternatives

<a id="markdown-generic" name="generic"></a>
### Generic

- [Cookiecutter](https://cookiecutter.readthedocs.io/), lot of templates, require python + pip + install dependencies on system (automatic)
- [Cookiecutter — Similar projects](https://cookiecutter.readthedocs.io/en/latest/readme.html#similar-projects)
- [sethyuan/fgen](https://github.com/sethyuan/fgen): A file generator library to be used to generate project structures, file templates and/or snippets. Templates are based on mustache. require nodejs
- [project_init](https://crates.io/crates/project_init) in rust, use mustache for templating but I have some issues with it (project template creation not obvious, github only, plus few bug) I could contributes but I have incompatible requirements (and would like to create my own since a long time).
- [skeleton](https://crates.io/crates/skeleton), good idea but no template file, more like a script.
- [porteurbars](https://crates.io/crates/porteurbars), very similar but I discover it too late.

<a id="markdown-specialized" name="specialized"></a>
### Specialized

specilazed to a platform, build tool,...

- [The web's scaffolding tool for modern webapps | Yeoman](http://yeoman.io/), nodejs ecosystem
- [JHipster - Generate your Spring Boot + Angular/React applications!](https://www.jhipster.tech/) require java, dedicated to java web ecosystem, optionnated template (not generic)
- [Giter8](http://www.foundweekends.org/giter8/) require java + [Conscript](http://www.foundweekends.org/conscript/index.html)
- [Typesafe activator](https://developer.lightbend.com/start/), require java, target scala ecosystem
- [Maven – Archetypes](https://maven.apache.org/guides/introduction/introduction-to-archetypes.html) require java + maven, target maven ecosystem
- [cargo-generate](https://github.com/ashleygwilliams/cargo-generate), limited capabilities, target rust/cargo ecosystem
