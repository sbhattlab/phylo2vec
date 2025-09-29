# JOSS submission

## Compiling the submission

The easiest way to compile the submission is to use Docker. In the project's root directory, run:

```bash
docker run --rm \
    --volume $PWD/joss:/data \
    --user $(id -u):$(id -g) \
    --env JOURNAL=joss \
    openjournals/inara
```

or:

```bash
docker run --rm -it \
    -v $PWD:/data \
    -u $(id -u):$(id -g) \
    openjournals/inara \
    -o pdf,crossref,preprint,jats \
    joss/paper.md
```

For more details, see the [JOSS Paper Format guidelines](https://joss.readthedocs.io/en/latest/paper.html).
```sudo``` rights may be required to execute these commands.

## Checking the word count

To double-check the word count, you may run:

```ruby
ruby joss/src/wordcount.rb
```

The script is inspired by [OpenJournals' buffy service](https://github.com/openjournals/buffy)
